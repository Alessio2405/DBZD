from __future__ import annotations

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from model.dbzd import DBZDModel, build_model
from model.fusion import ResidualZoneFusion
from dbzd.diagnostics import alpha_gradient_diagnostics


def _tiny_model(arm: str):
    torch.manual_seed(123)
    model = build_model(
        model_name="tiny",
        fallback_model=None,
        arm=arm,
        vocab_size=128,
        max_length=64,
        fork_layers=2,
    )
    model.eval()
    return model


def test_fusion_identity_is_bit_identical_when_disabled() -> None:
    torch.manual_seed(5)
    fusion = ResidualZoneFusion(hidden_size=16)
    generation = torch.randn(2, 7, 16)
    zones = torch.randn(2, 7, 16)
    fused, modulation = fusion(generation, zones, enabled=False)
    assert torch.equal(fused, generation)
    assert torch.equal(modulation, torch.ones_like(generation))


def test_branch_b_is_causal() -> None:
    model = _tiny_model("dbzd_full")
    original = torch.tensor([[4, 5, 6, 7, 8, 9]], dtype=torch.long)
    perturbed = original.clone()
    perturbed[0, 4:] = torch.tensor([20, 21])
    mask = torch.ones_like(original)
    with torch.no_grad():
        first = model(original, attention_mask=mask).zone_logits
        second = model(perturbed, attention_mask=mask).zone_logits
    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6, rtol=0.0)


def test_parameter_count_equal_across_arms() -> None:
    arms = ("baseline_matched", "multitask", "dbzd_full", "dbzd_stopgrad")
    counts = [_tiny_model(arm).total_parameter_count() for arm in arms]
    assert len(set(counts)) == 1


def test_gate_off_model_logits_match_plain_generation_head() -> None:
    model = _tiny_model("baseline_matched")
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids, attention_mask=mask, return_hidden=True)
        expected = model.lm_head(output.generation_hidden)
    assert torch.equal(output.logits, expected)


def test_next_token_projection_matches_full_forward() -> None:
    model = _tiny_model("dbzd_full")
    input_ids = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long)
    last_indices = torch.tensor([3, 2], dtype=torch.long)
    with torch.no_grad():
        full = model(input_ids, attention_mask=mask).logits
        compact = model.next_token_logits(input_ids, mask, last_indices)
    expected = torch.stack([full[0, 3], full[1, 2]])
    assert torch.allclose(compact, expected, atol=1e-6, rtol=0.0)


def test_stopgrad_lm_path_does_not_train_zone_tail() -> None:
    model = _tiny_model("dbzd_stopgrad")
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    output = model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        zone_labels=torch.tensor([[0, 1, 2, 3]]),
    )
    assert output.lm_loss is not None
    objective = output.lm_loss + model.gamma_reg * output.regularization_loss
    gradients = torch.autograd.grad(
        objective,
        list(model.zone_layers.parameters()),
        allow_unused=True,
    )
    assert all(
        gradient is None or torch.count_nonzero(gradient) == 0
        for gradient in gradients
    )


def test_dbzd_full_alpha_receives_lm_gradient_and_updates() -> None:
    model = _tiny_model("dbzd_full")
    input_ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids,
        "zone_labels": torch.tensor([[0, 1, 2, 3]]),
    }
    diagnostics = alpha_gradient_diagnostics(
        model,
        batch,
        device=torch.device("cpu"),
    )
    assert diagnostics["alpha_lm_gradient"] != 0.0
    assert diagnostics["alpha_total_gradient"] != 0.0

    optimizer = torch.optim.AdamW([model.fusion.alpha], lr=0.01, weight_decay=0.0)
    before = model.fusion.alpha.detach().clone()
    output = model(**batch)
    assert output.loss is not None
    output.loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.fusion.alpha.detach())


def test_gpt_neox_fallback_adapter_is_causal() -> None:
    config = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=64,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = DBZDModel(
        GPTNeoXForCausalLM(config),
        arm="dbzd_full",
        fork_layers=2,
    ).eval()
    original = torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long)
    perturbed = original.clone()
    perturbed[0, 4] = 20
    with torch.no_grad():
        first = model(original, attention_mask=torch.ones_like(original)).zone_logits
        second = model(
            perturbed, attention_mask=torch.ones_like(perturbed)
        ).zone_logits
    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6, rtol=0.0)
