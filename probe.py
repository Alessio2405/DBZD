from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from datagen.tokenizer import load_tokenizer
from dbzd.config import ExperimentConfig
from dbzd.data import CausalLMCollator, JSONLTokenDataset
from dbzd.utils import seed_everything, select_device
from model.dbzd import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linear probes for DBZD checkpoints.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-train-tokens", type=int, default=50_000)
    parser.add_argument("--max-test-tokens", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _load_run(run_dir: Path) -> tuple[ExperimentConfig, str, int]:
    payload = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    arm = str(payload.pop("arm"))
    seed = int(payload.pop("seed"))
    return ExperimentConfig(**payload), arm, seed


def _load_model(
    run_dir: Path,
    config: ExperimentConfig,
    arm: str,
    tokenizer: Any,
    device: torch.device,
):
    source_path = run_dir / "model_source.txt"
    model_name = (
        source_path.read_text(encoding="utf-8").strip()
        if source_path.exists()
        else config.model_name
    )
    model = build_model(
        model_name=model_name,
        fallback_model=None,
        arm=arm,
        vocab_size=len(tokenizer),
        max_length=config.max_length,
        fork_layers=config.fork_layers,
        num_zones=config.num_zones,
        lambda_zone=config.lambda_zone,
        gamma_reg=config.gamma_reg,
        alpha_init=config.alpha_init,
    )
    checkpoint_path = run_dir / "model_final.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint_latest.pt"
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def _extract(
    model,
    dataset: JSONLTokenDataset,
    collator: CausalLMCollator,
    *,
    device: torch.device,
    batch_size: int,
    token_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    trunk_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for batch in loader:
        shared_hidden, generation_hidden, _ = model.branch_hidden_states(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        labels = batch["zone_labels"]
        valid = labels.ge(0)
        trunk_parts.append(shared_hidden.detach().cpu()[valid].float().numpy())
        branch_parts.append(
            generation_hidden.detach().cpu()[valid].float().numpy()
        )
        label_parts.append(labels[valid].numpy())

    trunk = np.concatenate(trunk_parts)
    branch = np.concatenate(branch_parts)
    labels = np.concatenate(label_parts)
    if len(labels) > token_limit:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(labels), size=token_limit, replace=False)
        trunk, branch, labels = trunk[selected], branch[selected], labels[selected]
    return trunk, branch, labels


def run_probe(
    *,
    run_dir: str | Path,
    max_train_tokens: int = 50_000,
    max_test_tokens: int = 100_000,
    batch_size: int = 8,
    seed: int = 1234,
) -> Path:
    run_path = Path(run_dir)
    config, arm, run_seed = _load_run(run_path)
    seed_everything(seed)
    device = select_device()
    data_dir = Path(config.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer_name = (
        config.tokenizer_name
        or metadata.get("tokenizer", {}).get("name")
        or config.model_name
    )
    tokenizer = load_tokenizer(
        tokenizer_name,
        data_dir=data_dir,
        frozen_simple=True,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs a pad token for probing")
    model = _load_model(run_path, config, arm, tokenizer, device)
    collator = CausalLMCollator(int(tokenizer.pad_token_id), config.max_length)
    val_dataset = JSONLTokenDataset(data_dir / "val.jsonl")
    test_dataset = JSONLTokenDataset(data_dir / "test.jsonl")

    val_trunk, val_branch, val_labels = _extract(
        model,
        val_dataset,
        collator,
        device=device,
        batch_size=batch_size,
        token_limit=max_train_tokens,
        seed=seed,
    )
    test_trunk, test_branch, test_labels = _extract(
        model,
        test_dataset,
        collator,
        device=device,
        batch_size=batch_size,
        token_limit=max_test_tokens,
        seed=seed + 1,
    )

    results: dict[str, float] = {}
    for name, train_features, test_features in (
        ("trunk", val_trunk, test_trunk),
        ("branch_a", val_branch, test_branch),
    ):
        classifier = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=seed,
        )
        classifier.fit(train_features, val_labels)
        predictions = classifier.predict(test_features)
        results[f"{name}_macro_f1"] = float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        )

    payload = {
        "arm": arm,
        "seed": run_seed,
        "probe_seed": seed,
        "val_tokens": int(len(val_labels)),
        "test_tokens": int(len(test_labels)),
        **results,
    }
    output_path = run_path / "probe_summary.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"{arm} seed {run_seed}: trunk F1={results['trunk_macro_f1']:.3f}, "
        f"branch A F1={results['branch_a_macro_f1']:.3f}"
    )
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    run_probe(
        run_dir=args.run_dir,
        max_train_tokens=args.max_train_tokens,
        max_test_tokens=args.max_test_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
