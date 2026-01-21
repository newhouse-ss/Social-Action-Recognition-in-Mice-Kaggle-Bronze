# MABe Mouse Behavior Detection — Structured Repo Starter

This repository is a **refactoring starter** created from Kaggle notebooks contained in the uploaded zip.
Goal: make the solution **reviewable**, **modular**, and **reproducible** (aligned with typical Japan IT interview expectations).

## What is included
- Original notebooks under `notebooks/original/` (kept intact)
- A first-pass extraction of key code blocks into `src/mabe/`:
  - `metrics/` scoring functions
  - `gbm/` feature engineering + model factories
  - `ensemble/` ensembling utilities
- A raw port of the 7th-place **low-RAM test inference** cell as a script under `scripts/`

## Next refactor steps (recommended)
1. Replace Kaggle hard-coded paths with config (`configs/*.yaml`)
2. Make data loading explicit (`data_dir`, `meta.csv`, `tracking.parquet`)
3. Add a deterministic split generator + CV runner
4. Add unit tests for metrics and feature transforms

## Quick sanity check (local)
```bash
python -m compileall src scripts
```
