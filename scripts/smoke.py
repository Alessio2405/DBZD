from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _safe_remove(root: Path, target: Path) -> None:
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise RuntimeError(f"Refusing to remove path outside workspace: {resolved_target}")
    if target.exists():
        shutil.rmtree(target)


def _run(root: Path, *arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "smoke"
    run_root = root / "runs" / "smoke"
    _safe_remove(root, data_dir)
    _safe_remove(root, run_root)

    _run(
        root,
        "-m",
        "datagen",
        "--output-dir",
        str(data_dir),
        "--tokenizer",
        "simple",
        "--n",
        "24",
        "--seed",
        "1234",
    )
    _run(
        root,
        "train.py",
        "--config",
        "configs/smoke.yaml",
        "--arm",
        "dbzd_full",
        "--seed",
        "42",
    )
    # Exercise checkpoint restoration. With max_steps already reached this
    # performs no extra optimizer update, but reloads all mandatory state.
    _run(
        root,
        "train.py",
        "--config",
        "configs/smoke.yaml",
        "--arm",
        "dbzd_full",
        "--seed",
        "42",
        "--resume",
    )
    run_dir = run_root / "dbzd_full_s42"
    _run(
        root,
        "probe.py",
        "--run-dir",
        str(run_dir),
        "--max-train-tokens",
        "1000",
        "--max-test-tokens",
        "1000",
        "--batch-size",
        "2",
    )
    _run(root, "analysis.py", "--runs-dir", str(run_root))

    required = [
        data_dir / "train.jsonl",
        data_dir / "val.jsonl",
        data_dir / "test.jsonl",
        run_dir / "metrics.csv",
        run_dir / "checkpoint_latest.pt",
        run_dir / "summary.json",
        run_dir / "probe_summary.json",
        run_root / "analysis" / "aggregate_table.csv",
        run_root / "analysis" / "entropy_by_zone.png",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Smoke pipeline missed outputs: {missing}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["global_step"] != 3:
        raise AssertionError("Resume smoke unexpectedly changed optimizer step count")
    print("SMOKE PASS: data -> train -> resume -> probe -> analysis", flush=True)


if __name__ == "__main__":
    main()

