from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from datagen.tokenizer import load_tokenizer
from dbzd.config import ExperimentConfig, save_config
from dbzd.data import CausalLMCollator, JSONLTokenDataset
from dbzd.diagnostics import evaluate_loader, gradient_cosine, greedy_answer_accuracy
from dbzd.utils import (
    append_metric,
    autocast_context,
    capture_rng_state,
    cosine_warmup_lambda,
    git_hash,
    resolve_precision,
    restore_rng_state,
    seed_everything,
    select_device,
)
from model.dbzd import ARM_SETTINGS, DBZDModel, build_model
from model.fusion import clamp_fusion_alpha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one DBZD Phase 0 arm.")
    parser.add_argument("--arm", required=True, choices=tuple(ARM_SETTINGS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_loader(
    dataset: JSONLTokenDataset,
    collator: CausalLMCollator,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _checkpoint_payload(
    *,
    model: DBZDModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Any,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "rng_state": capture_rng_state(),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    model: DBZDModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Any,
) -> tuple[int, int, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload.get("scaler", {}))
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
    return (
        int(payload.get("epoch", 0)),
        int(payload.get("batch_in_epoch", 0)),
        int(payload.get("global_step", 0)),
    )


def _run_evaluation(
    *,
    model: DBZDModel,
    val_loader: DataLoader,
    test_dataset: JSONLTokenDataset,
    tokenizer: Any,
    config: ExperimentConfig,
    device: torch.device,
    global_step: int,
    epoch: float,
    metrics_path: Path,
    precision: str,
) -> dict[str, float]:
    metrics = evaluate_loader(
        model,
        val_loader,
        device=device,
        max_batches=config.max_eval_batches,
        precision=precision,
    )
    first_batch = next(iter(val_loader))
    metrics["gradient_cosine"] = gradient_cosine(
        model,
        first_batch,
        device=device,
        max_parameters=config.gradient_cosine_params,
        precision=precision,
    )
    metrics["answer_accuracy"] = greedy_answer_accuracy(
        model,
        test_dataset.records,
        tokenizer,
        device=device,
        max_length=config.max_length,
        max_new_tokens=config.answer_max_new_tokens,
        limit=config.answer_eval_examples,
        batch_size=config.answer_batch_size,
        precision=precision,
    )
    metrics.update(
        {
            "global_step": global_step,
            "epoch": epoch,
            "split": "val",
        }
    )
    append_metric(metrics_path, metrics)
    model.train()
    return metrics


def run_training(
    *,
    arm: str,
    seed: int,
    config: ExperimentConfig,
    resume: bool = False,
) -> Path:
    seed_everything(seed)
    device = select_device()
    precision = resolve_precision(device, config.precision)
    data_dir = Path(config.data_dir)
    run_dir = Path(config.run_root) / f"{arm}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "checkpoint_latest.pt"

    save_config(config, run_dir / "resolved_config.yaml", arm, seed)
    (run_dir / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad_token_id")

    train_dataset = JSONLTokenDataset(data_dir / "train.jsonl")
    val_dataset = JSONLTokenDataset(data_dir / "val.jsonl")
    test_dataset = JSONLTokenDataset(data_dir / "test.jsonl")
    collator = CausalLMCollator(int(pad_token_id), config.max_length)
    val_loader = _make_loader(
        val_dataset,
        collator,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=seed,
        num_workers=config.num_workers,
    )
    test_loader = _make_loader(
        test_dataset,
        collator,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=seed,
        num_workers=config.num_workers,
    )

    model = build_model(
        model_name=config.model_name,
        fallback_model=config.fallback_model,
        arm=arm,
        vocab_size=len(tokenizer),  # type: ignore[arg-type]
        max_length=config.max_length,
        fork_layers=config.fork_layers,
        num_zones=config.num_zones,
        lambda_zone=config.lambda_zone,
        gamma_reg=config.gamma_reg,
        alpha_init=config.alpha_init,
    ).to(device)
    (run_dir / "model_source.txt").write_text(
        model.source_model_name + "\n", encoding="utf-8"
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batches_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / config.grad_accum_steps)
    planned_steps = optimizer_steps_per_epoch * config.epochs
    total_steps = min(planned_steps, config.max_steps or planned_steps)
    warmup_steps = round(total_steps * config.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        functools.partial(
            cosine_warmup_lambda,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )
    scaler = _make_scaler(device.type == "cuda" and precision == "fp16")

    start_epoch = 0
    start_batch = 0
    global_step = 0
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requested but {checkpoint_path} is missing")
        start_epoch, start_batch, global_step = _load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        model.to(device)
        print(
            f"Resumed {arm} seed {seed} at epoch {start_epoch}, "
            f"batch {start_batch}, step {global_step}."
        )

    print(
        f"Training {arm} seed {seed} on {device} ({precision}); "
        f"{model.total_parameter_count():,} parameters."
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stopped_early = False
    last_epoch = start_epoch
    last_batch = start_batch

    for epoch in range(start_epoch, config.epochs):
        train_loader = _make_loader(
            train_dataset,
            collator,
            batch_size=config.batch_size,
            shuffle=True,
            seed=seed + epoch,
            num_workers=config.num_workers,
        )
        for batch_index, batch in enumerate(train_loader):
            if config.max_steps is not None and global_step >= config.max_steps:
                stopped_early = True
                break
            if epoch == start_epoch and batch_index < start_batch:
                continue
            last_epoch = epoch
            last_batch = batch_index + 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            zone_labels = batch["zone_labels"].to(device)

            with autocast_context(device, precision):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    zone_labels=zone_labels,
                )
                if output.loss is None:
                    raise RuntimeError("Training forward did not return a loss")
                scaled_loss = output.loss / config.grad_accum_steps

            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % config.grad_accum_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            clamp_fusion_alpha(model)
            scheduler.step()
            global_step += 1

            if global_step % config.log_every_steps == 0:
                append_metric(
                    metrics_path,
                    {
                        "global_step": global_step,
                        "epoch": epoch + (batch_index + 1) / len(train_loader),
                        "split": "train",
                        "learning_rate": scheduler.get_last_lr()[0],
                        "total_loss": float(output.loss.detach().float().item()),
                        "lm_loss": (
                            float(output.lm_loss.detach().float().item())
                            if output.lm_loss is not None
                            else ""
                        ),
                        "zone_loss": (
                            float(output.zone_loss.detach().float().item())
                            if output.zone_loss is not None
                            else ""
                        ),
                        "reg_loss": float(
                            output.regularization_loss.detach().float().item()
                        ),
                    },
                )

            if global_step % config.eval_every_steps == 0:
                eval_metrics = _run_evaluation(
                    model=model,
                    val_loader=val_loader,
                    test_dataset=test_dataset,
                    tokenizer=tokenizer,
                    config=config,
                    device=device,
                    global_step=global_step,
                    epoch=epoch + (batch_index + 1) / len(train_loader),
                    metrics_path=metrics_path,
                    precision=precision,
                )
                print(
                    f"step={global_step} val_lm={eval_metrics['lm_loss']:.4f} "
                    f"zone_f1={eval_metrics['zone_f1']:.3f} "
                    f"answer_acc={eval_metrics['answer_accuracy']:.3f}"
                )

            if global_step % config.save_every_steps == 0:
                _save_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        global_step=global_step,
                    ),
                )

            if config.max_steps is not None and global_step >= config.max_steps:
                stopped_early = True
                break

        start_batch = 0
        if stopped_early:
            break
        last_epoch = epoch + 1
        last_batch = 0
        _save_checkpoint(
            checkpoint_path,
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                batch_in_epoch=0,
                global_step=global_step,
            ),
        )

    _save_checkpoint(
        checkpoint_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=last_epoch,
            batch_in_epoch=last_batch,
            global_step=global_step,
        ),
    )

    val_metrics = _run_evaluation(
        model=model,
        val_loader=val_loader,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        config=config,
        device=device,
        global_step=global_step,
        epoch=float(last_epoch),
        metrics_path=metrics_path,
        precision=precision,
    )
    test_metrics = evaluate_loader(
        model,
        test_loader,
        device=device,
        max_batches=config.max_eval_batches,
        precision=precision,
    )
    test_first_batch = next(iter(test_loader))
    test_metrics["gradient_cosine"] = gradient_cosine(
        model,
        test_first_batch,
        device=device,
        max_parameters=config.gradient_cosine_params,
        precision=precision,
    )
    test_metrics["answer_accuracy"] = greedy_answer_accuracy(
        model,
        test_dataset.records,
        tokenizer,
        device=device,
        max_length=config.max_length,
        max_new_tokens=config.answer_max_new_tokens,
        limit=config.answer_eval_examples,
        batch_size=config.answer_batch_size,
        precision=precision,
    )
    append_metric(
        metrics_path,
        {
            **test_metrics,
            "global_step": global_step,
            "epoch": float(last_epoch),
            "split": "test",
        },
    )
    completed = not stopped_early and last_epoch >= config.epochs
    final_model_path = run_dir / "model_final.pt"
    if completed:
        _save_checkpoint(
            final_model_path,
            {
                "model": model.state_dict(),
                "arm": arm,
                "seed": seed,
                "global_step": global_step,
                "model_source": model.source_model_name,
            },
        )
    summary = {
        "arm": arm,
        "seed": seed,
        "global_step": global_step,
        "device": str(device),
        "precision": precision,
        "parameter_count": model.total_parameter_count(),
        "model_source": model.source_model_name,
        "completed": completed,
        "git_hash": git_hash(),
        "val": val_metrics,
        "test": test_metrics,
        "config": config.to_dict(),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if completed and config.compact_completed_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Finished {arm} seed {seed}; summary: {summary_path}")
    return summary_path


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_yaml(args.config).update(
        data_dir=args.data_dir,
        run_root=args.run_root,
        max_steps=args.max_steps,
    )
    run_training(arm=args.arm, seed=args.seed, config=config, resume=args.resume)


if __name__ == "__main__":
    main()
