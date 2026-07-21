from __future__ import annotations

import csv
import json

from scripts.recompute_taxonomy import recompute_run


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_offline_taxonomy_reclassification_requires_and_updates_full_files(
    tmp_path,
) -> None:
    run_dir = tmp_path / "baseline_matched_s42"
    run_dir.mkdir()
    summary = {
        "best_step": 1250,
        "selection": {"global_step": 1250},
        "last_val": {"global_step": 1250},
        "val": {"global_step": 1250},
        "test": {"global_step": 1250, "answer_eval_count": 2},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    fieldnames = [
        "global_step",
        "split",
        "answer_accuracy",
        "answer_eval_count",
    ]
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "global_step": 1250,
                    "split": "val",
                    "answer_accuracy": 0,
                    "answer_eval_count": 2,
                },
                {
                    "global_step": 1250,
                    "split": "val_best",
                    "answer_accuracy": 0,
                    "answer_eval_count": 2,
                },
                {
                    "global_step": 1250,
                    "split": "test_best",
                    "answer_accuracy": 0,
                    "answer_eval_count": 2,
                },
            ]
        )
    generations = [
        {
            "decoded_generation": "Step 1: Use 2 and 4. The answer is 2",
            "gold_answer": 24,
            "expected_operands": [2, 4],
            "error_category": "WRONG_OPERANDS",
        },
        {
            "decoded_generation": "Step 1: Use 2 and 4. The answer is 25.",
            "gold_answer": 24,
            "expected_operands": [2, 4],
            "error_category": "ARITHMETIC_ERROR",
        },
    ]
    _write_jsonl(run_dir / "generations_step_001250.jsonl", generations)
    _write_jsonl(run_dir / "generations_best_final.jsonl", generations)

    assert recompute_run(run_dir)
    updated = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert updated["answer_error_taxonomy"] == {
        "PARSE_FAIL": 1,
        "WRONG_OPERANDS": 0,
        "ARITHMETIC_ERROR": 1,
    }
    assert updated["test"]["answer_parse_fail_count"] == 1
    saved = [
        json.loads(line)
        for line in (run_dir / "generations_step_001250.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert saved[0]["parsed_answer"] is None
    assert saved[0]["error_category"] == "PARSE_FAIL"


def test_offline_taxonomy_refuses_partial_generation_files(tmp_path) -> None:
    run_dir = tmp_path / "dbzd_full_s42"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"test": {"answer_eval_count": 2}}), encoding="utf-8"
    )
    (run_dir / "metrics.csv").write_text(
        "global_step,split,answer_eval_count\n1,test_best,2\n",
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "generations_best_final.jsonl",
        [
            {
                "decoded_generation": "The answer is 2",
                "gold_answer": 24,
                "expected_operands": [2, 4],
            }
        ],
    )
    original = (run_dir / "summary.json").read_text(encoding="utf-8")
    assert not recompute_run(run_dir)
    assert (run_dir / "summary.json").read_text(encoding="utf-8") == original
