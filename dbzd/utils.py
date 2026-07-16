from __future__ import annotations

import csv
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch

from datagen.generator import ZONE_NAMES


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(device: torch.device, requested: str) -> str:
    if device.type != "cuda":
        return "fp32"
    if requested in {"fp16", "bf16", "fp32"}:
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            return "fp16"
        return requested
    if requested != "auto":
        raise ValueError("precision must be one of auto, fp16, bf16, fp32")
    major, _ = torch.cuda.get_device_capability()
    return "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16"


def autocast_context(device: torch.device, precision: str) -> ContextManager[Any]:
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-a-git-repository"


def cosine_warmup_lambda(
    current_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = (current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


METRIC_FIELDS = [
    "timestamp",
    "global_step",
    "epoch",
    "split",
    "learning_rate",
    "total_loss",
    "lm_loss",
    "zone_loss",
    "reg_loss",
    "zone_f1",
    "gradient_cosine",
    "answer_accuracy",
    "gate_mean",
    "gate_std",
    "entropy_mean",
]
for _zone_id, _zone_name in enumerate(ZONE_NAMES):
    METRIC_FIELDS.extend(
        [
            f"gate_mean_z{_zone_id}",
            f"gate_std_z{_zone_id}",
            f"entropy_z{_zone_id}",
        ]
    )


def append_metric(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    normalized = {field: row.get(field, "") for field in METRIC_FIELDS}
    normalized["timestamp"] = row.get("timestamp", time.time())
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

