from __future__ import annotations

import argparse
from pathlib import Path

from .generator import generate_dataset
from .tokenizer import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DBZD Phase 0 synthetic data.")
    parser.add_argument("--output-dir", default="data/phase0")
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--train-n", type=int, default=40_000)
    parser.add_argument("--val-n", type=int, default=2_000)
    parser.add_argument("--test-n", type=int, default=2_000)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Tiny mode: total examples, split approximately 90/5/5.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = load_tokenizer(args.tokenizer, data_dir=args.output_dir)
    metadata = generate_dataset(
        Path(args.output_dir),
        tokenizer,
        tokenizer_name=args.tokenizer,
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
        n=args.n,
        seed=args.seed,
    )
    print(f"Wrote dataset to {Path(args.output_dir).resolve()}")
    print(f"Counts: {metadata['counts']}")


if __name__ == "__main__":
    main()

