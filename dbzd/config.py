from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    experiment_revision: str = "phase0_final_r3"
    model_name: str = "HuggingFaceTB/SmolLM-135M"
    fallback_model: str | None = "EleutherAI/pythia-160m"
    tokenizer_name: str | None = None
    data_dir: str = "data/phase0"
    run_root: str = "runs"
    fork_layers: int = 2
    num_zones: int = 7
    max_length: int = 512
    alpha_init: float = 0.3
    lambda_zone: float = 0.5
    gamma_reg: float = 0.001
    epochs: float = 1.5
    batch_size: int = 4
    eval_batch_size: int = 8
    grad_accum_steps: int = 8
    learning_rate: float = 1.25e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_steps: int | None = None
    eval_every_steps: int = 250
    save_every_steps: int = 250
    log_every_steps: int = 20
    max_eval_batches: int | None = None
    answer_eval_examples: int = 512
    answer_batch_size: int = 32
    answer_max_new_tokens: int | None = None
    answer_length_percentile: float = 0.99
    answer_length_margin_tokens: int = 20
    generation_sample_count: int = 10
    gradient_cosine_params: int = 8
    precision: str = "auto"
    num_workers: int = 2
    compact_completed_checkpoint: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid = {field.name for field in fields(cls)}
        unknown = set(payload) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**payload)

    def update(self, **overrides: Any) -> "ExperimentConfig":
        values = asdict(self)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return ExperimentConfig(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_config(config: ExperimentConfig, path: str | Path, arm: str, seed: int) -> None:
    payload = config.to_dict()
    payload["arm"] = arm
    payload["seed"] = seed
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
