from __future__ import annotations

import csv
import json

import pytest

from scripts.validate_alpha import validate_dbzd_full_alpha


def _write_run(tmp_path, gradient: str, alpha: str = "0.298"):
    run_dir = tmp_path / "dbzd_full_s42"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "arm": "dbzd_full",
                "config": {"alpha_init": 0.3},
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "global_step",
                "split",
                "alpha",
                "alpha_lm_gradient",
                "alpha_total_gradient",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "global_step": 250,
                "split": "val",
                "alpha": alpha,
                "alpha_lm_gradient": gradient,
                "alpha_total_gradient": gradient,
            }
        )
    return run_dir


def test_validate_alpha_accepts_finite_gradient_and_updated_alpha(tmp_path) -> None:
    trajectory = validate_dbzd_full_alpha(_write_run(tmp_path, "-0.001"))
    assert trajectory[0]["alpha_lm_gradient"] == -0.001


@pytest.mark.parametrize(
    ("gradient", "alpha"),
    [("nan", "0.298"), ("-0.001", "0.3")],
)
def test_validate_alpha_rejects_broken_trajectory(
    tmp_path, gradient: str, alpha: str
) -> None:
    with pytest.raises(RuntimeError):
        validate_dbzd_full_alpha(_write_run(tmp_path, gradient, alpha))
