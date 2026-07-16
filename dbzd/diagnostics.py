from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from datagen.generator import ZONE_NAMES
from datagen.tokenizer import TokenizerLike
from dbzd.utils import autocast_context
from model.dbzd import DBZDModel


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        float_values = values.detach().float()
        self.count += int(float_values.numel())
        self.total += float(float_values.sum().item())
        self.square_total += float(float_values.square().sum().item())

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        if not self.count:
            return float("nan")
        variance = max(0.0, self.square_total / self.count - self.mean**2)
        return math.sqrt(variance)


def _macro_f1(confusion: torch.Tensor) -> float:
    scores: list[float] = []
    for zone_id in range(confusion.shape[0]):
        true_positive = float(confusion[zone_id, zone_id])
        false_positive = float(confusion[:, zone_id].sum() - true_positive)
        false_negative = float(confusion[zone_id, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def gradient_cosine(
    model: DBZDModel,
    batch: dict[str, Any],
    *,
    device: torch.device,
    max_parameters: int,
    precision: str = "fp32",
) -> float:
    parameters = [
        parameter
        for parameter in model.shared_trunk_parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        return float("nan")
    if len(parameters) > max_parameters:
        indices = np.linspace(0, len(parameters) - 1, max_parameters, dtype=int)
        parameters = [parameters[index] for index in indices]

    model.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        output = model(
            input_ids=batch["input_ids"][:1].to(device),
            attention_mask=batch["attention_mask"][:1].to(device),
            labels=batch["labels"][:1].to(device),
            zone_labels=batch["zone_labels"][:1].to(device),
        )
    if output.lm_loss is None or output.zone_loss is None:
        return float("nan")
    lm_grads = torch.autograd.grad(
        output.lm_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    zone_grads = torch.autograd.grad(
        output.zone_loss,
        parameters,
        allow_unused=True,
    )
    cosines: list[float] = []
    for lm_grad, zone_grad in zip(lm_grads, zone_grads):
        if lm_grad is None or zone_grad is None:
            continue
        lm_flat = lm_grad.detach().float().flatten()
        zone_flat = zone_grad.detach().float().flatten()
        if lm_flat.norm() == 0 or zone_flat.norm() == 0:
            continue
        cosines.append(float(torch.nn.functional.cosine_similarity(
            lm_flat, zone_flat, dim=0
        ).item()))
    model.zero_grad(set_to_none=True)
    return float(np.mean(cosines)) if cosines else float("nan")


@torch.no_grad()
def evaluate_loader(
    model: DBZDModel,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
    precision: str = "fp32",
) -> dict[str, float]:
    model.eval()
    loss_totals: dict[str, float] = defaultdict(float)
    batch_count = 0
    gate_overall = RunningMoments()
    entropy_overall = RunningMoments()
    gate_by_zone = [RunningMoments() for _ in ZONE_NAMES]
    entropy_by_zone = [RunningMoments() for _ in ZONE_NAMES]
    confusion = torch.zeros((len(ZONE_NAMES), len(ZONE_NAMES)), dtype=torch.long)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        with autocast_context(device, precision):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                zone_labels=batch["zone_labels"].to(device),
            )
        batch_count += 1
        if output.loss is not None:
            loss_totals["total_loss"] += float(output.loss.item())
        if output.lm_loss is not None:
            loss_totals["lm_loss"] += float(output.lm_loss.item())
        if output.zone_loss is not None:
            loss_totals["zone_loss"] += float(output.zone_loss.item())
        loss_totals["reg_loss"] += float(output.regularization_loss.item())

        zone_labels = batch["zone_labels"].to(device)
        valid_zone = zone_labels.ge(0)
        gate_values = output.modulation.float()
        gate_overall.update(gate_values[valid_zone])
        for zone_id in range(len(ZONE_NAMES)):
            gate_by_zone[zone_id].update(gate_values[zone_labels.eq(zone_id)])

        shifted_logits = output.logits[:, :-1].float()
        shifted_zones = zone_labels[:, 1:]
        valid_entropy = shifted_zones.ge(0)
        log_probs = torch.log_softmax(shifted_logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy_overall.update(entropy[valid_entropy])
        for zone_id in range(len(ZONE_NAMES)):
            entropy_by_zone[zone_id].update(entropy[shifted_zones.eq(zone_id)])

        predictions = output.zone_logits.argmax(dim=-1)
        for target, prediction in zip(
            zone_labels[valid_zone].detach().cpu(),
            predictions[valid_zone].detach().cpu(),
        ):
            confusion[int(target), int(prediction)] += 1

    metrics = {
        key: value / max(1, batch_count) for key, value in loss_totals.items()
    }
    metrics.update(
        {
            "zone_f1": _macro_f1(confusion),
            "gate_mean": gate_overall.mean,
            "gate_std": gate_overall.std,
            "entropy_mean": entropy_overall.mean,
        }
    )
    for zone_id in range(len(ZONE_NAMES)):
        metrics[f"gate_mean_z{zone_id}"] = gate_by_zone[zone_id].mean
        metrics[f"gate_std_z{zone_id}"] = gate_by_zone[zone_id].std
        metrics[f"entropy_z{zone_id}"] = entropy_by_zone[zone_id].mean
    return metrics


_ANSWER_PATTERN = re.compile(r"The\s+answer\s+is\s+(-?\d+)", re.IGNORECASE)


@torch.no_grad()
def greedy_answer_accuracy(
    model: DBZDModel,
    records: Iterable[dict[str, Any]],
    tokenizer: TokenizerLike,
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    limit: int,
    batch_size: int = 8,
    precision: str = "fp32",
) -> float:
    model.eval()
    correct = 0
    attempted = 0
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs a pad token for batched generation")
    selected = list(records)[:limit]
    for start in range(0, len(selected), batch_size):
        batch_records = selected[start : start + batch_size]
        prompts = [
            list(record["tokens"][: int(record["prompt_token_count"])])
            for record in batch_records
        ]
        sequences = [list(prompt) for prompt in prompts]
        generated = [[] for _ in batch_records]
        finished = [False for _ in batch_records]
        for _ in range(max_new_tokens):
            context_sequences = [sequence[-max_length:] for sequence in sequences]
            lengths = [len(sequence) for sequence in context_sequences]
            longest = max(lengths)
            padded_sequences = [
                sequence + [int(pad_token_id)] * (longest - len(sequence))
                for sequence in context_sequences
            ]
            context_masks = [
                [1] * len(sequence) + [0] * (longest - len(sequence))
                for sequence in context_sequences
            ]
            input_ids = torch.tensor(
                padded_sequences, dtype=torch.long, device=device
            )
            attention_mask = torch.tensor(
                context_masks, dtype=torch.long, device=device
            )
            with autocast_context(device, precision):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            for row, length in enumerate(lengths):
                if finished[row]:
                    continue
                token = int(output.logits[row, length - 1].argmax().item())
                sequences[row].append(token)
                generated[row].append(token)
                if eos_token_id is not None and token == eos_token_id:
                    finished[row] = True
            if all(finished):
                break
        for record, token_ids in zip(batch_records, generated):
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            match = _ANSWER_PATTERN.search(decoded)
            predicted = int(match.group(1)) if match else None
            correct += int(predicted == int(record["answer"]))
            attempted += 1
    return correct / attempted if attempted else float("nan")
