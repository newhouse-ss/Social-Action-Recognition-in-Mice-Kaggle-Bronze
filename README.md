# MABe Mouse Behavior Detection

Refactors the original `notebook/` code into clear, reusable modules while keeping the same pipeline for easier reading and GitHub presentation.

## Project Layout

- `mabe/`: core source code (data loading, feature engineering, training/inference, post-processing, evaluation)
- `scripts/`: CLI entry points (training + submission, offline evaluation)
- `docs/Mabe.md`: method overview and EDA notes
- `notebook/`: original notebooks (archived)

## Quick Start

1. Install dependencies
   - `pip install -r requirements.txt`
2. Prepare data
   - Uses `MABE_DATA_DIR` if set
   - Otherwise defaults to Kaggle path: `/kaggle/input/MABe-mouse-behavior-detection`
3. Generate a submission
   - `python scripts/make_submission.py --data-dir "path/to/MABe-mouse-behavior-detection" --output submission.csv`
4. Local evaluation (optional)
   - `python scripts/evaluate.py --solution train.csv --submission submission.csv`

## Notes

- Training/inference is time- and memory-intensive; Kaggle or GPU is recommended.
- Use `--no-gpu` to force CPU; `--n-samples` controls subsampling size.
