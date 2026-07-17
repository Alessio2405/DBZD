from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datagen.generator import ZONE_NAMES
from model.dbzd import ARM_SETTINGS

REPORT_METRICS = (
    "answer_accuracy",
    "answer_parse_fail_count",
    "answer_wrong_operands_count",
    "answer_arithmetic_error_count",
    "probe_trunk_f1",
    "probe_branch_a_f1",
    *(f"entropy_z{zone_id}" for zone_id in range(len(ZONE_NAMES))),
    *(f"gate_mean_z{zone_id}" for zone_id in range(len(ZONE_NAMES))),
    "gate_mean",
    "gate_std",
    "alpha",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate DBZD Phase 0 runs.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default=None)
    return parser


def _load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(runs_dir.glob("*_s*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        test = summary["test"]
        record: dict[str, Any] = {
            "arm": summary["arm"],
            "seed": summary["seed"],
            "run_dir": summary_path.parent,
            "answer_accuracy": test.get("answer_accuracy", float("nan")),
            "answer_parse_fail_count": test.get(
                "answer_parse_fail_count", float("nan")
            ),
            "answer_wrong_operands_count": test.get(
                "answer_wrong_operands_count", float("nan")
            ),
            "answer_arithmetic_error_count": test.get(
                "answer_arithmetic_error_count", float("nan")
            ),
            "gate_mean": test.get("gate_mean", float("nan")),
            "gate_std": test.get("gate_std", float("nan")),
            "alpha": test.get("alpha", float("nan")),
            "best_step": summary.get("best_step"),
        }
        for zone_id in range(len(ZONE_NAMES)):
            record[f"entropy_z{zone_id}"] = test.get(
                f"entropy_z{zone_id}", float("nan")
            )
            record[f"gate_mean_z{zone_id}"] = test.get(
                f"gate_mean_z{zone_id}", float("nan")
            )
        probe_path = summary_path.parent / "probe_summary.json"
        if probe_path.exists():
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            record["probe_trunk_f1"] = probe.get("trunk_macro_f1", float("nan"))
            record["probe_branch_a_f1"] = probe.get(
                "branch_a_macro_f1", float("nan")
            )
        else:
            record["probe_trunk_f1"] = float("nan")
            record["probe_branch_a_f1"] = float("nan")
        records.append(record)
    return records


def _aggregate(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["arm"]].append(record)
    result: dict[str, dict[str, tuple[float, float]]] = {}
    for arm, arm_records in grouped.items():
        result[arm] = {}
        result[arm]["_n"] = (float(len(arm_records)), 0.0)
        for metric in REPORT_METRICS:
            values = np.asarray(
                [float(record.get(metric, float("nan"))) for record in arm_records],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if not len(finite):
                result[arm][metric] = (float("nan"), float("nan"))
            else:
                result[arm][metric] = (
                    float(np.mean(finite)),
                    float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                )
    return result


def _format(mean_std: tuple[float, float]) -> str:
    mean, std = mean_std
    if not math.isfinite(mean):
        return "n/a"
    return f"{mean:.4f} +/- {std:.4f}"


def _write_table(
    aggregate: dict[str, dict[str, tuple[float, float]]],
    output_dir: Path,
) -> str:
    compact_metrics = [
        "answer_accuracy",
        "answer_parse_fail_count",
        "answer_wrong_operands_count",
        "answer_arithmetic_error_count",
        "probe_trunk_f1",
        "probe_branch_a_f1",
        "entropy_z3",
        "entropy_z6",
        "gate_mean",
        "gate_std",
        "alpha",
    ]
    lines = [
        "| arm | answer acc | parse fail | wrong operands | arithmetic error | probe trunk F1 | probe branch A F1 | "
        "Z3 entropy | Z6 entropy | gate mean | gate std | alpha |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows: list[dict[str, Any]] = []
    for arm in ARM_SETTINGS:
        if arm not in aggregate:
            continue
        values = [_format(aggregate[arm][metric]) for metric in compact_metrics]
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
        row: dict[str, Any] = {"arm": arm}
        for metric in REPORT_METRICS:
            row[f"{metric}_mean"], row[f"{metric}_std"] = aggregate[arm][metric]
        csv_rows.append(row)
    table = "\n".join(lines)
    (output_dir / "aggregate_table.md").write_text(table + "\n", encoding="utf-8")
    if csv_rows:
        with (output_dir / "aggregate_table.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    return table


def _bar_plot(
    aggregate: dict[str, dict[str, tuple[float, float]]],
    *,
    prefix: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    arms = [arm for arm in ARM_SETTINGS if arm in aggregate]
    if not arms:
        return
    x = np.arange(len(ZONE_NAMES))
    width = 0.8 / len(arms)
    fig, axis = plt.subplots(figsize=(12, 5))
    for arm_index, arm in enumerate(arms):
        means = [aggregate[arm][f"{prefix}{zone_id}"][0] for zone_id in x]
        axis.bar(
            x + (arm_index - (len(arms) - 1) / 2) * width,
            means,
            width,
            label=arm,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([f"Z{i}\n{name}" for i, name in enumerate(ZONE_NAMES)])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _training_curves(records: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    found = False
    for record in records:
        metrics_path = Path(record["run_dir"]) / "metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        train_rows = [row for row in all_rows if row["split"] == "train"]
        val_rows = [row for row in all_rows if row["split"] == "val"]
        if not train_rows and not val_rows:
            continue
        found = True
        label = f"{record['arm']}-s{record['seed']}"
        if val_rows:
            val_steps = [int(row["global_step"]) for row in val_rows]
            val_lm = [float(row["lm_loss"]) for row in val_rows]
            train_lm = [float(row["train_lm_loss"]) for row in val_rows]
            axes[0].plot(val_steps, train_lm, marker="o", label=f"{label} train")
            axes[0].plot(
                val_steps,
                val_lm,
                marker="s",
                linestyle="--",
                label=f"{label} val",
            )
            axes[1].plot(
                val_steps,
                [float(row["zone_loss"]) for row in val_rows],
                marker="s",
                linestyle="--",
                label=f"{label} val",
            )
        elif train_rows:
            steps = [int(row["global_step"]) for row in train_rows]
            axes[0].plot(
                steps,
                [float(row["lm_loss"]) for row in train_rows],
                alpha=0.65,
                label=f"{label} train",
            )
            axes[1].plot(
                steps,
                [float(row["zone_loss"]) for row in train_rows],
                alpha=0.65,
                label=f"{label} train",
            )
        if record.get("best_step") is not None:
            axes[0].axvline(
                int(record["best_step"]), color="black", alpha=0.18, linewidth=1
            )
    if not found:
        plt.close(fig)
        return
    axes[0].set_title("Train/validation LM loss (best step marked)")
    axes[1].set_title("Validation zone loss")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("loss")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _pooled_std(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt(a[1] ** 2 + b[1] ** 2)


def _verdict(
    aggregate: dict[str, dict[str, tuple[float, float]]],
) -> str:
    required = {"baseline_matched", "multitask", "dbzd_full", "dbzd_stopgrad"}
    if not required.issubset(aggregate):
        missing = ", ".join(sorted(required - set(aggregate)))
        return f"INCOMPLETE: missing arms for pre-registered verdict: {missing}."
    short_arms = [
        arm for arm in sorted(required) if int(aggregate[arm]["_n"][0]) < 3
    ]
    if short_arms:
        counts = ", ".join(
            f"{arm}={int(aggregate[arm]['_n'][0])}" for arm in short_arms
        )
        return f"INCOMPLETE: three seeds per arm are required ({counts})."

    full = aggregate["dbzd_full"]
    baseline = aggregate["baseline_matched"]
    multitask = aggregate["multitask"]
    stopgrad = aggregate["dbzd_stopgrad"]
    probe_advantage = (
        full["probe_trunk_f1"][0] - baseline["probe_trunk_f1"][0]
        > _pooled_std(full["probe_trunk_f1"], baseline["probe_trunk_f1"])
    )
    entropy_better = full["entropy_z6"][0] < baseline["entropy_z6"][0]
    accuracy_better = full["answer_accuracy"][0] > baseline["answer_accuracy"][0]
    pass_phase_1 = probe_advantage and (entropy_better or accuracy_better)

    multitask_gap = abs(
        multitask["probe_trunk_f1"][0] - full["probe_trunk_f1"][0]
    )
    null_2 = multitask_gap <= _pooled_std(
        multitask["probe_trunk_f1"], full["probe_trunk_f1"]
    )
    coupled = (
        full["probe_trunk_f1"][0] - stopgrad["probe_trunk_f1"][0]
        > _pooled_std(full["probe_trunk_f1"], stopgrad["probe_trunk_f1"])
    )
    return "\n".join(
        [
            f"PASS to Phase 1: {'YES' if pass_phase_1 else 'NO'}",
            f"Null #2 confirmed (multitask ≈ full): {'YES' if null_2 else 'NO'}",
            f"Coupled-gradient evidence: {'YES' if coupled else 'NO'}",
        ]
    )


def run_analysis(
    runs_dir: str | Path = "runs",
    output_dir: str | Path | None = None,
) -> Path:
    runs_path = Path(runs_dir)
    output_path = Path(output_dir) if output_dir else runs_path / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    records = _load_runs(runs_path)
    if not records:
        raise FileNotFoundError(
            f"No immediate *_s*/summary.json run directories found in {runs_path}"
        )
    aggregate = _aggregate(records)
    table = _write_table(aggregate, output_path)
    _bar_plot(
        aggregate,
        prefix="entropy_z",
        ylabel="predictive entropy",
        title="LM predictive entropy by target-token zone",
        output_path=output_path / "entropy_by_zone.png",
    )
    _bar_plot(
        aggregate,
        prefix="gate_mean_z",
        ylabel="mean modulation",
        title="Final gate mean by zone",
        output_path=output_path / "gate_mean_by_zone.png",
    )
    _training_curves(records, output_path / "training_curves.png")
    verdict = _verdict(aggregate)
    (output_path / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")
    print(table)
    print("\n" + verdict)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    run_analysis(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
