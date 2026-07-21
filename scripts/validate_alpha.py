from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def validate_dbzd_full_alpha(run_dir: str | Path) -> list[dict[str, float]]:
    path = Path(run_dir)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("arm") != "dbzd_full":
        raise ValueError(f"Expected dbzd_full, found {summary.get('arm')!r}")
    alpha_init = float((summary.get("config") or {}).get("alpha_init", 0.3))
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "val"]
    trajectory: list[dict[str, float]] = []
    for row in rows:
        try:
            trajectory.append(
                {
                    "step": float(row["global_step"]),
                    "alpha": float(row["alpha"]),
                    "alpha_lm_gradient": float(row["alpha_lm_gradient"]),
                    "alpha_total_gradient": float(row["alpha_total_gradient"]),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Missing alpha diagnostic at step {row.get('global_step')}"
            ) from error
    if not trajectory:
        raise RuntimeError("No validation alpha trajectory was recorded")
    non_finite = [
        point for point in trajectory if not math.isfinite(point["alpha_lm_gradient"])
    ]
    if non_finite:
        raise RuntimeError(f"dbzd_full alpha_lm_gradient is non-finite: {non_finite}")
    if not any(abs(point["alpha"] - alpha_init) > 1e-7 for point in trajectory):
        raise RuntimeError(
            f"dbzd_full alpha never moved from alpha_init={alpha_init}: {trajectory}"
        )
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if a dbzd_full run did not backpropagate into alpha."
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    trajectory = validate_dbzd_full_alpha(args.run_dir)
    print("dbzd_full alpha trajectory:")
    for point in trajectory:
        print(
            f"  step={int(point['step'])} alpha={point['alpha']:.8f} "
            f"lm_grad={point['alpha_lm_gradient']:+.6e} "
            f"total_grad={point['alpha_total_gradient']:+.6e}"
        )


if __name__ == "__main__":
    main()
