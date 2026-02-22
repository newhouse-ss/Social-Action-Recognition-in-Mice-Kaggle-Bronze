"""
Centralized configuration for the AWS Serverless ETL & EDA Pipeline.

All S3 paths, bucket names, and Spark settings are defined here so every
script in the pipeline reads from one source of truth.

For LOCAL simulation:
    - Set USE_LOCAL_FS = True  →  all S3 paths map to local directories.
    - Set USE_LOCAL_FS = False →  real boto3 / S3 calls are used.
"""

import os

# ── Toggle: local filesystem vs real S3 ──────────────────────────────────
USE_LOCAL_FS: bool = True

# ── S3 Bucket / Prefix ───────────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET", "mabe-mouse-behavior-datalake")
S3_RAW_PREFIX = "raw-zone"          # landing area for raw uploads
S3_CURATED_PREFIX = "curated-zone"  # cleaned / transformed parquet
S3_ATHENA_PREFIX = "athena-results" # query result spill (isolated)

# ── Local mirror paths (used when USE_LOCAL_FS=True) ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_RAW_DATA = os.path.join(PROJECT_ROOT, "data")          # existing data/
LOCAL_S3_ROOT  = os.path.join(PROJECT_ROOT, "s3_mock")       # simulated bucket

LOCAL_RAW_ZONE     = os.path.join(LOCAL_S3_ROOT, S3_RAW_PREFIX)
LOCAL_CURATED_ZONE = os.path.join(LOCAL_S3_ROOT, S3_CURATED_PREFIX)
LOCAL_ATHENA_ZONE  = os.path.join(LOCAL_S3_ROOT, S3_ATHENA_PREFIX)

# ── Dataset constants ────────────────────────────────────────────────────
SELF_ACTIONS = [
    "biteobject", "climb", "dig", "exploreobject", "freeze",
    "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom",
]
PAIR_ACTIONS = [
    "allogroom", "approach", "attack", "attemptmount", "avoid",
    "chase", "chaseattack", "defend", "disengage", "dominance",
    "dominancegroom", "dominancemount", "ejaculate", "escape",
    "flinch", "follow", "intromit", "mount", "reciprocalsniff",
    "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital",
    "submit", "tussle",
]
ALL_ACTIONS = sorted(set(SELF_ACTIONS + PAIR_ACTIONS))

# ── Spark ────────────────────────────────────────────────────────────────
SPARK_APP_NAME = "MABe-Glue-ETL"
SPARK_DRIVER_MEMORY = "4g"
