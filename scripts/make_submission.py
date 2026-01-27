from __future__ import annotations

import argparse

from mabe.config import SEED
from mabe.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models and create a submission.csv.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root directory of Kaggle dataset (defaults to MABE_DATA_DIR or Kaggle path).",
    )
    parser.add_argument("--output", default="submission.csv", help="Output CSV path.")
    parser.add_argument("--n-samples", type=int, default=1_500_000, help="Subsample size.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        n_samples=args.n_samples,
        use_gpu=False if args.no_gpu else None,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
