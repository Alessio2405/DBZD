from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .fusion import ResidualZoneFusion

ARM_SETTINGS: dict[str, dict[str, bool]] = {
    "baseline_matched": {"fusion": False, "stop_gradient": False},
    "multitask": {"fusion": False, "stop_gradient": False},
    "dbzd_full": {"fusion": True, "stop_gradient": False},
    "dbzd_stopgrad": {"fusion": True, "stop_gradient": True},
}


@dataclass
class DBZDOutput:
    logits: torch.Tensor
    zone_logits: torch.Tensor
    modulation: torch.Tensor
    lm_loss: torch.Tensor | None
    zone_loss: torch.Tensor | None
    regularization_loss: torch.Tensor
    loss: torch.Tensor | None
    shared_hidden: torch.Tensor | None = None
    generation_hidden: torch.Tensor | None = None
    zone_hidden: torch.Tensor | None = None


def _additive_causal_mask(
    attention_mask: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length = hidden_states.shape[:2]
    min_value = torch.finfo(hidden_states.dtype).min
    mask = torch.full(
        (sequence_length, sequence_length),
        min_value,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    mask = torch.triu(mask, diagonal=1)
    mask = mask.view(1, 1, sequence_length, sequence_length).expand(
        batch_size, 1, sequence_length, sequence_length
    )
    if attention_mask is not None:
        key_padding = attention_mask[:, None, None, :].eq(0)
        mask = mask.masked_fill(key_padding, min_value)
    return mask


def _unwrap_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported transformer layer output: {type(output)!r}")


class DBZDModel(nn.Module):
    """A shared causal trunk with forked generation and zone tails."""

    def __init__(
        self,
        causal_lm: nn.Module,
        *,
        arm: str,
        fork_layers: int = 2,
        num_zones: int = 7,
        lambda_zone: float = 0.5,
        gamma_reg: float = 0.01,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        if arm not in ARM_SETTINGS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {tuple(ARM_SETTINGS)}")

        self.arm = arm
        self.fork_layers = fork_layers
        self.num_zones = num_zones
        self.lambda_zone = 0.0 if arm == "baseline_matched" else lambda_zone
        self.gamma_reg = gamma_reg
        self.fusion_enabled = ARM_SETTINGS[arm]["fusion"]
        self.stop_gradient = ARM_SETTINGS[arm]["stop_gradient"]
        self.backbone_kind = self._detect_backbone(causal_lm)

        if self.backbone_kind == "llama":
            self.backbone = causal_lm.model
            self.lm_head = causal_lm.lm_head
            layers = self.backbone.layers
            hidden_size = int(causal_lm.config.hidden_size)
            final_norm = self.backbone.norm
        elif self.backbone_kind == "gpt_neox":
            self.backbone = causal_lm.gpt_neox
            self.lm_head = causal_lm.embed_out
            layers = self.backbone.layers
            hidden_size = int(causal_lm.config.hidden_size)
            final_norm = self.backbone.final_layer_norm
        else:  # pragma: no cover - guarded by _detect_backbone
            raise AssertionError(self.backbone_kind)

        if fork_layers < 1 or fork_layers >= len(layers):
            raise ValueError(
                f"fork_layers must be in [1, {len(layers) - 1}], got {fork_layers}"
            )
        self.fork_index = len(layers) - fork_layers
        self.zone_layers = nn.ModuleList(copy.deepcopy(list(layers[self.fork_index :])))
        self.zone_norm = copy.deepcopy(final_norm)
        self.zone_head = nn.Linear(hidden_size, num_zones)
        self.fusion = ResidualZoneFusion(hidden_size, alpha_init=alpha_init)
        self.fusion.set_trainable(self.fusion_enabled)

    @staticmethod
    def _detect_backbone(causal_lm: nn.Module) -> str:
        if hasattr(causal_lm, "model") and hasattr(causal_lm.model, "layers"):
            return "llama"
        if hasattr(causal_lm, "gpt_neox") and hasattr(causal_lm.gpt_neox, "layers"):
            return "gpt_neox"
        raise TypeError(
            "DBZD supports LlamaForCausalLM (SmolLM) and "
            "GPTNeoXForCausalLM (Pythia) backbones."
        )

    @property
    def hidden_size(self) -> int:
        return int(self.zone_head.in_features)

    def _run_llama(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone.embed_tokens(input_ids)
        sequence_length = input_ids.shape[1]
        cache_position = torch.arange(sequence_length, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask = _additive_causal_mask(attention_mask, hidden)
        position_embeddings = self.backbone.rotary_emb(hidden, position_ids)
        layers = self.backbone.layers

        for layer in layers[: self.fork_index]:
            hidden = _unwrap_layer_output(
                layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            )
        shared_hidden = hidden

        generation_hidden = shared_hidden
        for layer in layers[self.fork_index :]:
            generation_hidden = _unwrap_layer_output(
                layer(
                    generation_hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            )
        generation_hidden = self.backbone.norm(generation_hidden)

        zone_hidden = shared_hidden
        for layer in self.zone_layers:
            zone_hidden = _unwrap_layer_output(
                layer(
                    zone_hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            )
        zone_hidden = self.zone_norm(zone_hidden)
        return shared_hidden, generation_hidden, zone_hidden

    def _run_gpt_neox(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone.emb_dropout(self.backbone.embed_in(input_ids))
        sequence_length = input_ids.shape[1]
        cache_position = torch.arange(sequence_length, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask = _additive_causal_mask(attention_mask, hidden)
        position_embeddings = self.backbone.rotary_emb(hidden, position_ids)
        layers = self.backbone.layers

        def run_layer(layer: nn.Module, states: torch.Tensor) -> torch.Tensor:
            return _unwrap_layer_output(
                layer(
                    states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    head_mask=None,
                    use_cache=False,
                    layer_past=None,
                    output_attentions=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            )

        for layer in layers[: self.fork_index]:
            hidden = run_layer(layer, hidden)
        shared_hidden = hidden

        generation_hidden = shared_hidden
        for layer in layers[self.fork_index :]:
            generation_hidden = run_layer(layer, generation_hidden)
        generation_hidden = self.backbone.final_layer_norm(generation_hidden)

        zone_hidden = shared_hidden
        for layer in self.zone_layers:
            zone_hidden = run_layer(layer, zone_hidden)
        zone_hidden = self.zone_norm(zone_hidden)
        return shared_hidden, generation_hidden, zone_hidden

    def branch_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.backbone_kind == "llama":
            return self._run_llama(input_ids, attention_mask)
        return self._run_gpt_neox(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        zone_labels: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> DBZDOutput:
        shared_hidden, generation_hidden, zone_hidden = self.branch_hidden_states(
            input_ids, attention_mask
        )
        fused_hidden, modulation = self.fusion(
            generation_hidden,
            zone_hidden,
            enabled=self.fusion_enabled,
            stop_gradient=self.stop_gradient,
        )
        logits = self.lm_head(fused_hidden)
        zone_logits = self.zone_head(zone_hidden)

        lm_loss: torch.Tensor | None = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        zone_loss: torch.Tensor | None = None
        if zone_labels is not None:
            zone_loss = F.cross_entropy(
                zone_logits.view(-1, self.num_zones),
                zone_labels.view(-1),
                ignore_index=-100,
            )

        regularization_source = (
            self.fusion.modulation(zone_hidden.detach())
            if self.stop_gradient
            else modulation
        )
        regularization_loss = (regularization_source - 1.0).square().mean()
        total_loss: torch.Tensor | None = None
        if lm_loss is not None:
            total_loss = lm_loss + self.gamma_reg * regularization_loss
            if zone_loss is not None and self.lambda_zone != 0.0:
                total_loss = total_loss + self.lambda_zone * zone_loss

        return DBZDOutput(
            logits=logits,
            zone_logits=zone_logits,
            modulation=modulation,
            lm_loss=lm_loss,
            zone_loss=zone_loss,
            regularization_loss=regularization_loss,
            loss=total_loss,
            shared_hidden=shared_hidden if return_hidden else None,
            generation_hidden=generation_hidden if return_hidden else None,
            zone_hidden=zone_hidden if return_hidden else None,
        )

    def next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return only each sequence's final valid-position LM logits.

        Batched answer evaluation pads prompts to different lengths. Avoiding
        the vocabulary projection at every earlier position makes full-test
        greedy decoding substantially cheaper without changing predictions.
        """
        _, generation_hidden, zone_hidden = self.branch_hidden_states(
            input_ids, attention_mask
        )
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        generation_last = generation_hidden[rows, last_indices].unsqueeze(1)
        zone_last = zone_hidden[rows, last_indices].unsqueeze(1)
        fused_last, _ = self.fusion(
            generation_last,
            zone_last,
            enabled=self.fusion_enabled,
            stop_gradient=self.stop_gradient,
        )
        return self.lm_head(fused_last).squeeze(1)

    def shared_trunk_parameters(self) -> Iterator[nn.Parameter]:
        if self.backbone_kind == "llama":
            yield from self.backbone.embed_tokens.parameters()
        else:
            yield from self.backbone.embed_in.parameters()
        for layer in self.backbone.layers[: self.fork_index]:
            yield from layer.parameters()

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _tiny_causal_lm(vocab_size: int, max_length: int) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max(256, max_length),
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config)


def build_model(
    *,
    model_name: str,
    fallback_model: str | None,
    arm: str,
    vocab_size: int,
    max_length: int,
    fork_layers: int = 2,
    num_zones: int = 7,
    lambda_zone: float = 0.5,
    gamma_reg: float = 0.01,
    alpha_init: float = 0.1,
) -> DBZDModel:
    loaded_model_name = model_name
    if model_name == "tiny":
        causal_lm = _tiny_causal_lm(vocab_size, max_length)
    else:
        from transformers import AutoModelForCausalLM

        try:
            causal_lm = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception:
            if not fallback_model:
                raise
            print(f"Could not load {model_name}; falling back to {fallback_model}.")
            causal_lm = AutoModelForCausalLM.from_pretrained(fallback_model)
            loaded_model_name = fallback_model
        embedding_count = causal_lm.get_input_embeddings().num_embeddings
        if embedding_count != vocab_size:
            causal_lm.resize_token_embeddings(vocab_size)

    model = DBZDModel(
        causal_lm,
        arm=arm,
        fork_layers=fork_layers,
        num_zones=num_zones,
        lambda_zone=lambda_zone,
        gamma_reg=gamma_reg,
        alpha_init=alpha_init,
    )
    model.source_model_name = loaded_model_name
    return model
