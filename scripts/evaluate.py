from __future__ import annotations

import argparse

import pandas as pd

from mabe.metrics import score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a submission with MABe F-beta.")
    parser.add_argument("--solution", required=True, help="Solution CSV path.")
    parser.add_argument("--submission", required=True, help="Submission CSV path.")
    parser.add_argument("--row-id-col", default="row_id", help="Row id column name.")
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
