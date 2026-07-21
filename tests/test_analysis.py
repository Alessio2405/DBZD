from __future__ import annotations

import math

from analysis import _aggregate, _verdict, _write_error_taxonomy_table, _write_table


def test_baseline_zone_f1_and_alpha_are_reported_as_not_applicable(tmp_path) -> None:
    aggregate = _aggregate(
        [
            {
                "arm": "baseline_matched",
                "zone_f1": 0.14,
                "alpha": 0.3,
            },
            {
                "arm": "dbzd_full",
                "zone_f1": 0.97,
                "alpha": 0.28,
            },
        ]
    )
    assert math.isnan(aggregate["baseline_matched"]["zone_f1"][0])
    assert math.isnan(aggregate["baseline_matched"]["alpha"][0])
    assert aggregate["dbzd_full"]["zone_f1"][0] == 0.97
    assert aggregate["dbzd_full"]["alpha"][0] == 0.28

    table = _write_table(aggregate, tmp_path)
    baseline_row = next(
        line for line in table.splitlines() if line.startswith("| baseline_matched ")
    )
    assert "zone F1 — (lambda=0); alpha — (identity gate)" in baseline_row
    assert "—" in baseline_row


def test_error_taxonomy_fraction_table_compares_arms(tmp_path) -> None:
    aggregate = _aggregate(
        [
            {
                "arm": "baseline_matched",
                "answer_parse_fail_fraction": 0.10,
                "answer_wrong_operands_fraction": 0.70,
                "answer_arithmetic_error_fraction": 0.20,
            },
            {
                "arm": "multitask",
                "answer_parse_fail_fraction": 0.10,
                "answer_wrong_operands_fraction": 0.30,
                "answer_arithmetic_error_fraction": 0.60,
            },
        ]
    )
    table = _write_error_taxonomy_table(aggregate, tmp_path)
    assert "PARSE_FAIL fraction" in table
    assert "| baseline_matched | 0.1000 +/- 0.0000" in table
    assert "| multitask | 0.1000 +/- 0.0000 | 0.3000 +/- 0.0000" in table
    assert (tmp_path / "error_taxonomy_table.csv").exists()


def test_verdict_refuses_missing_probe_metrics() -> None:
    records = []
    for arm in ("baseline_matched", "multitask", "dbzd_full", "dbzd_stopgrad"):
        for seed in (42, 43, 44):
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "answer_accuracy": 0.5,
                    "entropy_z6": 1.0,
                    "probe_trunk_f1": float("nan"),
                    "probe_branch_a_f1": float("nan"),
                }
            )
    verdict = _verdict(_aggregate(records))
    assert verdict.startswith("INCOMPLETE: missing finite verdict metrics:")
