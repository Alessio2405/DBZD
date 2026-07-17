from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbzd.diagnostics import (
    ANSWER_ERROR_CATEGORIES,
    calculation_operands,
    classify_answer_error_from_operands,
    parse_answer,
)


COUNT_FIELDS = {
    "PARSE_FAIL": "answer_parse_fail_count",
    "WRONG_OPERANDS": "answer_wrong_operands_count",
    "ARITHMETIC_ERROR": "answer_arithmetic_error_count",
}
STEP_FILE_PATTERN = re.compile(r"generations_step_(\d+)\.jsonl$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute strict answer parsing and taxonomy from complete saved "
            "generation JSONL files without retraining."
        )
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _taxonomy(records: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy = {category: 0 for category in ANSWER_ERROR_CATEGORIES}
    correct = 0
    for record in records:
        decoded = str(record["decoded_generation"])
        gold = int(record["gold_answer"])
        predicted = parse_answer(decoded)
        if "calculation" in record:
            expected_operands = calculation_operands(record["calculation"])
        elif "expected_operands" in record:
            expected_operands = [int(value) for value in record["expected_operands"]]
        else:
            raise ValueError(
                "generation record lacks both calculation and expected_operands"
            )
        category, generated_numbers = classify_answer_error_from_operands(
            decoded,
            expected_operands,
            predicted,
            gold,
        )
        is_correct = predicted == gold
        correct += int(is_correct)
        if category is not None:
            taxonomy[category] += 1
        record.update(
            {
                "parsed_answer": predicted,
                "correct": is_correct,
                "error_category": category,
                "expected_operands": expected_operands,
                "generated_step_numbers": generated_numbers,
            }
        )
    count = len(records)
    return {
        "answer_accuracy": correct / count if count else float("nan"),
        "answer_eval_count": count,
        "answer_correct_count": correct,
        **{field: taxonomy[category] for category, field in COUNT_FIELDS.items()},
        "taxonomy": taxonomy,
    }


def _apply_metrics(target: dict[str, Any], result: dict[str, Any]) -> None:
    for field in (
        "answer_accuracy",
        "answer_eval_count",
        "answer_correct_count",
        *COUNT_FIELDS.values(),
    ):
        target[field] = result[field]


def _expected_count(row: dict[str, Any], label: str) -> int:
    value = row.get("answer_eval_count")
    if value in (None, ""):
        raise ValueError(f"{label} has no answer_eval_count")
    return int(float(value))


def recompute_run(run_dir: Path, *, dry_run: bool = False) -> bool:
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    final_path = run_dir / "generations_best_final.jsonl"
    if not summary_path.exists() or not metrics_path.exists() or not final_path.exists():
        print(f"SKIP {run_dir}: summary, metrics, or final generations missing")
        return False

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        metric_rows = list(reader)

    jobs: list[tuple[Path, int, int | None]] = []
    for path in sorted(run_dir.glob("generations_step_*.jsonl")):
        match = STEP_FILE_PATTERN.search(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        candidates = [
            row
            for row in metric_rows
            if row.get("split") == "val" and int(row.get("global_step") or -1) == step
        ]
        if not candidates:
            raise ValueError(f"{path} has no matching val metrics row")
        jobs.append((path, _expected_count(candidates[-1], path.name), step))

    final_expected = int(summary["test"]["answer_eval_count"])
    jobs.append((final_path, final_expected, None))

    loaded: dict[Path, list[dict[str, Any]]] = {}
    incomplete: list[str] = []
    for path, expected, _ in jobs:
        records = _read_jsonl(path)
        loaded[path] = records
        if len(records) != expected:
            incomplete.append(f"{path.name}: saved {len(records)}, expected {expected}")
    if incomplete:
        print(f"INCOMPLETE {run_dir}: taxonomy was not changed")
        for detail in incomplete:
            print("  " + detail)
        return False

    changed = False
    results: dict[Path, dict[str, Any]] = {}
    for path, _, _ in jobs:
        before = [record.get("error_category") for record in loaded[path]]
        result = _taxonomy(loaded[path])
        results[path] = result
        after = [record.get("error_category") for record in loaded[path]]
        changed = changed or before != after

    for path, _, step in jobs:
        result = results[path]
        if step is None:
            for row in metric_rows:
                if row.get("split") == "test_best":
                    _apply_metrics(row, result)
            _apply_metrics(summary["test"], result)
            summary["answer_error_taxonomy"] = result["taxonomy"]
            continue

        for row in metric_rows:
            if (
                row.get("split") in {"val", "val_best"}
                and int(row.get("global_step") or -1) == step
            ):
                _apply_metrics(row, result)
        if int(summary.get("best_step", -1)) == step:
            for key in ("selection", "val"):
                if isinstance(summary.get(key), dict):
                    _apply_metrics(summary[key], result)
        last_val = summary.get("last_val")
        if isinstance(last_val, dict) and int(last_val.get("global_step", -1)) == step:
            _apply_metrics(last_val, result)

    if dry_run:
        print(f"DRY RUN {run_dir}: complete files; classification_changed={changed}")
        return True

    for path, _, _ in jobs:
        _write_jsonl(path, loaded[path])
    for field in (
        "answer_correct_count",
        *COUNT_FIELDS.values(),
    ):
        if field not in fieldnames:
            fieldnames.append(field)
    temporary_metrics = metrics_path.with_suffix(".csv.tmp")
    with temporary_metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metric_rows)
    temporary_metrics.replace(metrics_path)
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    print(f"UPDATED {run_dir}: classification_changed={changed}")
    return True


def main() -> None:
    args = build_parser().parse_args()
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = [path.parent for path in sorted(Path(args.runs_dir).glob("*_s*/summary.json"))]
    if not run_dirs:
        raise FileNotFoundError("No run summaries found")
    successful = 0
    for run_dir in run_dirs:
        try:
            successful += int(recompute_run(run_dir, dry_run=args.dry_run))
        except (KeyError, TypeError, ValueError) as error:
            print(f"ERROR {run_dir}: {error}")
    if successful != len(run_dirs):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
