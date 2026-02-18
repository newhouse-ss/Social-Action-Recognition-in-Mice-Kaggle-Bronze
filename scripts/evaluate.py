"""Evaluate a submission CSV against a ground-truth solution CSV.

Usage::

    python scripts/evaluate.py --solution data/train.csv --submission submission.csv
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.metrics import score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a submission with the F-beta metric.")
    parser.add_argument("--solution", required=True, help="Ground-truth solution CSV path.")
    parser.add_argument("--submission", required=True, help="Submission CSV path.")
    parser.add_argument("--row-id-col", default="row_id", help="Row-ID column name.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for F-beta.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solution = pd.read_csv(args.solution)
    submission = pd.read_csv(args.submission)
    metric = score(solution, submission, args.row_id_col, beta=args.beta)
    print(f"F{args.beta:g} score: {metric:.6f}")


if __name__ == "__main__":
    main()
