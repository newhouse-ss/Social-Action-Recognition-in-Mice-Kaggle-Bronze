"""
Step 1 — Data Lake Storage: Upload raw datasets to Amazon S3 (Raw Zone).

This script uses **boto3** to upload the local Kaggle dataset to an S3
bucket, mirroring the workflow you would use in production:

    local data/
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    ├── train_tracking/{lab_id}/{video_id}.parquet
    ├── test_tracking/{lab_id}/{video_id}.parquet
    └── train_annotation/{lab_id}/{video_id}.parquet

        ──▶  s3://{bucket}/raw-zone/
              ├── metadata/train.csv
              ├── metadata/test.csv
              ├── metadata/sample_submission.csv
              ├── train_tracking/lab_id=<lab>/<video_id>.parquet
              ├── test_tracking/lab_id=<lab>/<video_id>.parquet
              └── train_annotation/lab_id=<lab>/<video_id>.parquet

Usage (local simulation):
    python aws_pipeline/s3_upload.py

Usage (real S3):
    export S3_BUCKET=my-bucket
    export USE_LOCAL_FS=false
    python aws_pipeline/s3_upload.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# -- Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aws_pipeline.config import (
    LOCAL_RAW_DATA,
    LOCAL_RAW_ZONE,
    S3_BUCKET,
    S3_RAW_PREFIX,
    USE_LOCAL_FS,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _upload_file_s3(local_path: str, s3_key: str, s3_client) -> None:
    """Upload a single file to S3."""
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)


def _copy_file_local(local_path: str, dest_path: str) -> None:
    """Copy a file into the local S3-mock directory tree."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(local_path, dest_path)


# ── main ─────────────────────────────────────────────────────────────────

def upload_raw_data() -> None:
    """Walk the local data/ directory and upload everything to the Raw Zone."""

    data_root = Path(LOCAL_RAW_DATA)
    if not data_root.exists():
        print(f"[ERROR] Local data directory not found: {data_root}")
        sys.exit(1)

    # Optionally init boto3
    s3_client = None
    if not USE_LOCAL_FS:
        import boto3
        s3_client = boto3.client("s3")
        print(f"[INFO] Uploading to s3://{S3_BUCKET}/{S3_RAW_PREFIX}/")
    else:
        os.makedirs(LOCAL_RAW_ZONE, exist_ok=True)
        print(f"[INFO] Local-FS mode → copying to {LOCAL_RAW_ZONE}")

    uploaded = 0

    # ── 1. CSV metadata files ────────────────────────────────────────
    for csv_name in ["train.csv", "test.csv", "sample_submission.csv"]:
        src = data_root / csv_name
        if not src.exists():
            continue
        s3_key = f"{S3_RAW_PREFIX}/metadata/{csv_name}"
        if USE_LOCAL_FS:
            _copy_file_local(str(src), os.path.join(LOCAL_RAW_ZONE, "metadata", csv_name))
        else:
            _upload_file_s3(str(src), s3_key, s3_client)
        uploaded += 1
        print(f"  ✓ {csv_name}")

    # ── 2. Tracking & annotation parquets (Hive-partitioned by lab_id) ─
    for subdir in ["train_tracking", "test_tracking", "train_annotation"]:
        subdir_path = data_root / subdir
        if not subdir_path.exists():
            continue
        for lab_dir in sorted(subdir_path.iterdir()):
            if not lab_dir.is_dir():
                continue
            lab_id = lab_dir.name
            parquet_files = list(lab_dir.glob("*.parquet"))
            for pf in parquet_files:
                # Hive-style partition: subdir/lab_id=<lab>/<file>.parquet
                s3_key = f"{S3_RAW_PREFIX}/{subdir}/lab_id={lab_id}/{pf.name}"
                if USE_LOCAL_FS:
                    dest = os.path.join(LOCAL_RAW_ZONE, subdir, f"lab_id={lab_id}", pf.name)
                    _copy_file_local(str(pf), dest)
                else:
                    _upload_file_s3(str(pf), s3_key, s3_client)
                uploaded += 1

            print(f"  ✓ {subdir}/lab_id={lab_id}/ ({len(parquet_files)} files)")

    print(f"\n[DONE] Uploaded {uploaded} objects to Raw Zone.")


if __name__ == "__main__":
    upload_raw_data()
