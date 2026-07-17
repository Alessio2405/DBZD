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

from datagen.generator import DATAGEN_SCHEMA_VERSION
from datagen.tokenizer import load_tokenizer
from dbzd.config import ExperimentConfig, save_config
from dbzd.data import CausalLMCollator, JSONLTokenDataset
from dbzd.diagnostics import (
    evaluate_loader,
    gold_completion_token_budget,
    gradient_cosine,
    greedy_answer_evaluation,
)
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


def _load_model_checkpoint(path: Path, model: DBZDModel) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    return payload


def _write_generations(
    path: Path,
    generations: list[dict[str, Any]],
    *,
    label: str,
    sample_count: int,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for generation in generations:
            handle.write(json.dumps(generation, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(generations)} generations to {path.name}.")
    print(f"Generation samples ({label}):")
    for index, sample in enumerate(generations[:sample_count], start=1):
        decoded = str(sample["decoded_generation"]).replace("\n", " ")
        print(
            f"  [{index:02d}] gold={sample['gold_answer']} "
            f"parsed={sample['parsed_answer']} correct={sample['correct']} "
            f"error={sample.get('error_category')}\n"
            f"       {decoded}"
        )


def _write_gate_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = ["zone_id,zone,gate_mean,gate_std"]
    print("\nBest-checkpoint gate statistics by zone:")
    print("  zone                         mean       std")
    from datagen.generator import ZONE_NAMES

    for zone_id, zone_name in enumerate(ZONE_NAMES):
        mean = float(metrics[f"gate_mean_z{zone_id}"])
        std = float(metrics[f"gate_std_z{zone_id}"])
        lines.append(f"{zone_id},{zone_name},{mean:.8f},{std:.8f}")
        print(f"  Z{zone_id} {zone_name:<24} {mean:>8.5f}  {std:>8.5f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    run_dir: Path,
    precision: str,
    train_lm_loss: float,
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
    answer_result = greedy_answer_evaluation(
        model,
        test_dataset.records,
        tokenizer,
        device=device,
        max_length=config.max_length,
        max_new_tokens=config.answer_max_new_tokens,
        limit=config.answer_eval_examples,
        batch_size=config.answer_batch_size,
        precision=precision,
        sample_count=config.generation_sample_count,
    )
    metrics["answer_accuracy"] = float(answer_result["accuracy"])
    metrics["answer_eval_count"] = int(answer_result["count"])
    metrics["answer_correct_count"] = int(answer_result["correct"])
    metrics["answer_parse_fail_count"] = int(
        answer_result["taxonomy"]["PARSE_FAIL"]
    )
    metrics["answer_wrong_operands_count"] = int(
        answer_result["taxonomy"]["WRONG_OPERANDS"]
    )
    metrics["answer_arithmetic_error_count"] = int(
        answer_result["taxonomy"]["ARITHMETIC_ERROR"]
    )
    metrics["train_lm_loss"] = train_lm_loss
    metrics.update(
        {
            "global_step": global_step,
            "epoch": epoch,
            "split": "val",
        }
    )
    append_metric(metrics_path, metrics)
    _write_generations(
        run_dir / f"generations_step_{global_step:06d}.jsonl",
        answer_result["generations"],
        label=f"step {global_step}; n={answer_result['count']}",
        sample_count=config.generation_sample_count,
    )
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
    if run_dir.exists() and not resume:
        generated_names = {
            "metrics.csv",
            "checkpoint_latest.pt",
            "checkpoint_best.pt",
            "best_metrics.json",
            "model_final.pt",
            "summary.json",
            "probe_summary.json",
            "gate_per_zone.csv",
        }
        for path in run_dir.iterdir():
            if path.name in generated_names or path.name.startswith("generations_"):
                if path.is_file():
                    path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    best_metrics_path = run_dir / "best_metrics.json"

    (run_dir / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("generator_schema_version") != DATAGEN_SCHEMA_VERSION:
        raise RuntimeError(
            f"Dataset schema {metadata.get('generator_schema_version')} is stale; "
            f"revision {config.experiment_revision} requires datagen schema "
            f"{DATAGEN_SCHEMA_VERSION}. Regenerate {data_dir}."
        )
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
    generation_budget = gold_completion_token_budget(
        test_dataset.records,
        eos_token_id=tokenizer.eos_token_id,
        percentile=config.answer_length_percentile,
        margin_tokens=config.answer_length_margin_tokens,
    )
    configured_generation_limit = config.answer_max_new_tokens
    if configured_generation_limit is None:
        effective_generation_limit = int(generation_budget["max_new_tokens"])
        config = config.update(answer_max_new_tokens=effective_generation_limit)
    else:
        effective_generation_limit = int(configured_generation_limit)
    generation_budget["effective_max_new_tokens"] = effective_generation_limit
    generation_budget["configured_override"] = configured_generation_limit
    save_config(config, run_dir / "resolved_config.yaml", arm, seed)
    computed_generation_limit = int(generation_budget["max_new_tokens"])
    budget_message = (
        "Answer generation budget: "
        f"p{config.answer_length_percentile * 100:g}="
        f"{generation_budget['percentile_tokens']} + "
        f"{config.answer_length_margin_tokens} margin = "
        f"{computed_generation_limit} tokens"
    )
    if configured_generation_limit is not None:
        budget_message += f"; configured override = {effective_generation_limit}"
    print(budget_message + ".")
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
    planned_steps = math.ceil(optimizer_steps_per_epoch * config.epochs)
    total_steps = min(
        planned_steps,
        config.max_steps if config.max_steps is not None else planned_steps,
    )
    limited_by_max_steps = (
        config.max_steps is not None and config.max_steps < planned_steps
    )
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
    best_val_lm = float("inf")
    best_step = -1
    best_epoch = 0.0
    best_selection_metrics: dict[str, Any] = {}
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
        if best_metrics_path.exists() and best_checkpoint_path.exists():
            best_state = json.loads(best_metrics_path.read_text(encoding="utf-8"))
            best_val_lm = float(best_state["val_lm"])
            best_step = int(best_state["global_step"])
            best_epoch = float(best_state["epoch"])
            best_selection_metrics = dict(best_state["selection_metrics"])

    print(
        f"Training {arm} seed {seed} on {device} ({precision}); "
        f"{model.total_parameter_count():,} parameters."
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stopped_early = False
    reached_training_target = global_step >= total_steps
    last_epoch = start_epoch
    last_batch = start_batch
    train_lm_total = 0.0
    train_lm_count = 0
    last_train_lm = float("nan")
    last_eval_step = -1
    last_eval_metrics: dict[str, float] | None = None
    optimizer_updates_this_invocation = 0

    for epoch in range(start_epoch, math.ceil(config.epochs)):
        train_loader = _make_loader(
            train_dataset,
            collator,
            batch_size=config.batch_size,
            shuffle=True,
            seed=seed + epoch,
            num_workers=config.num_workers,
        )
        for batch_index, batch in enumerate(train_loader):
            if global_step >= total_steps:
                reached_training_target = True
                stopped_early = limited_by_max_steps
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

            if output.lm_loss is not None:
                train_lm_total += float(output.lm_loss.detach().float().item())
                train_lm_count += 1

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
            optimizer_updates_this_invocation += 1

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
                        "alpha": float(
                            model.fusion.alpha.detach().clamp(0.0, 1.0).item()
                        ),
                    },
                )

            if global_step % config.eval_every_steps == 0:
                # The answer diagnostic is intentionally large; persist all
                # resumable state before entering that long evaluation.
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
                current_train_lm = (
                    train_lm_total / train_lm_count
                    if train_lm_count
                    else last_train_lm
                )
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
                    run_dir=run_dir,
                    precision=precision,
                    train_lm_loss=current_train_lm,
                )
                last_train_lm = current_train_lm
                train_lm_total = 0.0
                train_lm_count = 0
                last_eval_step = global_step
                last_eval_metrics = eval_metrics
                if float(eval_metrics["lm_loss"]) < best_val_lm:
                    best_val_lm = float(eval_metrics["lm_loss"])
                    best_step = global_step
                    best_epoch = float(eval_metrics["epoch"])
                    best_selection_metrics = dict(eval_metrics)
                    _save_checkpoint(
                        best_checkpoint_path,
                        {
                            "model": model.state_dict(),
                            "global_step": best_step,
                            "epoch": best_epoch,
                            "val_lm": best_val_lm,
                            "selection_metrics": best_selection_metrics,
                            "model_source": model.source_model_name,
                        },
                    )
                    best_metrics_path.write_text(
                        json.dumps(
                            {
                                "global_step": best_step,
                                "epoch": best_epoch,
                                "val_lm": best_val_lm,
                                "selection_metrics": best_selection_metrics,
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    print(f"New best checkpoint: step={best_step} val_lm={best_val_lm:.4f}")
                print(
                    f"step={global_step} train_lm={current_train_lm:.4f} "
                    f"val_lm={eval_metrics['lm_loss']:.4f} "
                    f"zone_f1={eval_metrics['zone_f1']:.3f} "
                    f"answer_acc={eval_metrics['answer_accuracy']:.3f} "
                    f"(n={int(eval_metrics['answer_eval_count'])}) "
                    f"alpha={eval_metrics['alpha']:.4f}"
                )
                print(
                    "  answer errors: "
                    f"PARSE_FAIL={int(eval_metrics['answer_parse_fail_count'])} "
                    "WRONG_OPERANDS="
                    f"{int(eval_metrics['answer_wrong_operands_count'])} "
                    "ARITHMETIC_ERROR="
                    f"{int(eval_metrics['answer_arithmetic_error_count'])}"
                )

            if (
                global_step % config.save_every_steps == 0
                and global_step % config.eval_every_steps != 0
            ):
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

            if global_step >= total_steps:
                reached_training_target = True
                stopped_early = limited_by_max_steps
                break

        start_batch = 0
        if reached_training_target:
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

    final_epoch_progress = (
        float(last_epoch)
        if last_batch == 0
        else float(last_epoch) + last_batch / batches_per_epoch
    )
    current_train_lm = (
        train_lm_total / train_lm_count if train_lm_count else last_train_lm
    )
    if optimizer_updates_this_invocation == 0 and best_selection_metrics:
        last_eval_metrics = dict(best_selection_metrics)
        last_eval_step = global_step
    if last_eval_step != global_step or last_eval_metrics is None:
        last_eval_metrics = _run_evaluation(
            model=model,
            val_loader=val_loader,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            config=config,
            device=device,
            global_step=global_step,
            epoch=final_epoch_progress,
            metrics_path=metrics_path,
            run_dir=run_dir,
            precision=precision,
            train_lm_loss=current_train_lm,
        )
        last_eval_step = global_step

    if float(last_eval_metrics["lm_loss"]) < best_val_lm:
        best_val_lm = float(last_eval_metrics["lm_loss"])
        best_step = global_step
        best_epoch = float(last_eval_metrics["epoch"])
        best_selection_metrics = dict(last_eval_metrics)
        _save_checkpoint(
            best_checkpoint_path,
            {
                "model": model.state_dict(),
                "global_step": best_step,
                "epoch": best_epoch,
                "val_lm": best_val_lm,
                "selection_metrics": best_selection_metrics,
                "model_source": model.source_model_name,
            },
        )
        best_metrics_path.write_text(
            json.dumps(
                {
                    "global_step": best_step,
                    "epoch": best_epoch,
                    "val_lm": best_val_lm,
                    "selection_metrics": best_selection_metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"New best checkpoint: step={best_step} val_lm={best_val_lm:.4f}")

    if not best_checkpoint_path.exists():
        raise RuntimeError("Training finished without producing a best checkpoint")
    best_payload = _load_model_checkpoint(best_checkpoint_path, model)
    model.to(device).eval()
    best_step = int(best_payload["global_step"])
    best_epoch = float(best_payload["epoch"])
    best_val_lm = float(best_payload["val_lm"])
    best_selection_metrics = dict(best_payload["selection_metrics"])
    print(
        f"Selected best checkpoint at step {best_step} "
        f"(epoch {best_epoch:.3f}, val_lm={best_val_lm:.4f})."
    )

    # All reported final representation, entropy, gate, and answer metrics are
    # recomputed after loading the selected best checkpoint.
    val_metrics = evaluate_loader(
        model,
        val_loader,
        device=device,
        max_batches=config.max_eval_batches,
        precision=precision,
    )
    val_metrics["gradient_cosine"] = gradient_cosine(
        model,
        next(iter(val_loader)),
        device=device,
        max_parameters=config.gradient_cosine_params,
        precision=precision,
    )
    val_metrics.update(
        {
            "train_lm_loss": best_selection_metrics.get("train_lm_loss"),
            "answer_accuracy": best_selection_metrics.get("answer_accuracy"),
            "answer_eval_count": best_selection_metrics.get("answer_eval_count"),
            "answer_correct_count": best_selection_metrics.get(
                "answer_correct_count"
            ),
            "answer_parse_fail_count": best_selection_metrics.get(
                "answer_parse_fail_count"
            ),
            "answer_wrong_operands_count": best_selection_metrics.get(
                "answer_wrong_operands_count"
            ),
            "answer_arithmetic_error_count": best_selection_metrics.get(
                "answer_arithmetic_error_count"
            ),
            "global_step": best_step,
            "epoch": best_epoch,
            "split": "val_best",
        }
    )
    append_metric(metrics_path, val_metrics)

    test_metrics = evaluate_loader(
        model,
        test_loader,
        device=device,
        max_batches=config.max_eval_batches,
        precision=precision,
    )
    test_metrics["gradient_cosine"] = gradient_cosine(
        model,
        next(iter(test_loader)),
        device=device,
        max_parameters=config.gradient_cosine_params,
        precision=precision,
    )
    final_answer_result = greedy_answer_evaluation(
        model,
        test_dataset.records,
        tokenizer,
        device=device,
        max_length=config.max_length,
        max_new_tokens=config.answer_max_new_tokens,
        limit=None,
        batch_size=config.answer_batch_size,
        precision=precision,
        sample_count=config.generation_sample_count,
    )
    test_metrics.update(
        {
            "answer_accuracy": float(final_answer_result["accuracy"]),
            "answer_eval_count": int(final_answer_result["count"]),
            "answer_correct_count": int(final_answer_result["correct"]),
            "answer_parse_fail_count": int(
                final_answer_result["taxonomy"]["PARSE_FAIL"]
            ),
            "answer_wrong_operands_count": int(
                final_answer_result["taxonomy"]["WRONG_OPERANDS"]
            ),
            "answer_arithmetic_error_count": int(
                final_answer_result["taxonomy"]["ARITHMETIC_ERROR"]
            ),
            "global_step": best_step,
            "epoch": best_epoch,
            "split": "test_best",
        }
    )
    append_metric(metrics_path, test_metrics)
    _write_generations(
        run_dir / "generations_best_final.jsonl",
        final_answer_result["generations"],
        label=f"best checkpoint, full test n={final_answer_result['count']}",
        sample_count=config.generation_sample_count,
    )
    _write_gate_report(run_dir / "gate_per_zone.csv", test_metrics)
    print(
        "\nFull-test answer taxonomy: "
        + ", ".join(
            f"{category}={count}"
            for category, count in final_answer_result["taxonomy"].items()
        )
    )

    completed = not limited_by_max_steps and global_step >= planned_steps
    final_model_path = run_dir / "model_final.pt"
    if completed:
        _save_checkpoint(
            final_model_path,
            {
                "model": model.state_dict(),
                "arm": arm,
                "seed": seed,
                "global_step": best_step,
                "best_val_lm": best_val_lm,
                "model_source": model.source_model_name,
            },
        )
    summary = {
        "arm": arm,
        "seed": seed,
        "global_step": global_step,
        "best_step": best_step,
        "best_epoch": best_epoch,
        "best_val_lm": best_val_lm,
        "device": str(device),
        "precision": precision,
        "parameter_count": model.total_parameter_count(),
        "model_source": model.source_model_name,
        "completed": completed,
        "git_hash": git_hash(),
        "selection": best_selection_metrics,
        "last_val": last_eval_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "answer_error_taxonomy": final_answer_result["taxonomy"],
        "answer_generation_budget": generation_budget,
        "generation_samples": "generations_best_final.jsonl",
        "gate_report": "gate_per_zone.csv",
        "config": config.to_dict(),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if completed and config.compact_completed_checkpoint:
        for compacted_path in (checkpoint_path, best_checkpoint_path):
            if compacted_path.exists():
                compacted_path.unlink()
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
