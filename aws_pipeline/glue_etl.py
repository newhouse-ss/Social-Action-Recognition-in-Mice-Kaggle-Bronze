"""
Step 2 — Serverless ETL Pipeline (AWS Glue Simulation).

This PySpark script simulates an **AWS Glue Job** that:

  1. Reads raw tracking & annotation Parquet files from the S3 Raw Zone.
  2. Cleans and standardizes the data:
     - Resolves cross-lab coordinate inconsistencies (pixel → centimeter).
     - Handles missing keypoints via forward-fill + backward-fill.
     - Adds velocity features (dx/dt, dy/dt) per body part.
  3. Joins metadata with tracking data using SparkSQL.
  4. Computes annotation summary statistics (class distribution, duration).
  5. Writes curated datasets back to the S3 Curated Zone in **partitioned
     Parquet** format (partitioned by ``lab_id``), ready for Athena queries.

Usage (local simulation with local PySpark):
    python aws_pipeline/glue_etl.py

Usage (real AWS Glue):
    Upload this script to S3 and configure a Glue Job pointing to it.
    Set the Glue Job parameters for --S3_BUCKET, etc.
"""

from __future__ import annotations

import os
import sys

# -- Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from aws_pipeline.config import (
    ALL_ACTIONS,
    LOCAL_CURATED_ZONE,
    LOCAL_RAW_ZONE,
    SELF_ACTIONS,
    SPARK_APP_NAME,
    SPARK_DRIVER_MEMORY,
    USE_LOCAL_FS,
)


# =====================================================================
# Spark Session
# =====================================================================

def create_spark_session() -> SparkSession:
    """Create a local SparkSession that mirrors an AWS Glue environment."""
    return (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master("local[*]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.parquet.mergeSchema", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )


# =====================================================================
# Step 2a — Ingest raw data from Raw Zone
# =====================================================================

def read_metadata(spark: SparkSession, raw_zone: str) -> DataFrame:
    """Read train.csv and test.csv from the Raw Zone metadata folder."""
    train_path = os.path.join(raw_zone, "metadata", "train.csv")
    test_path = os.path.join(raw_zone, "metadata", "test.csv")

    train_df = spark.read.option("header", True).option("inferSchema", True).csv(train_path)
    test_df = spark.read.option("header", True).option("inferSchema", True).csv(test_path)

    # Add a split column for downstream partitioning
    train_df = train_df.withColumn("split", F.lit("train"))
    test_df = test_df.withColumn("split", F.lit("test"))

    return train_df, test_df


def read_tracking(spark: SparkSession, raw_zone: str, subdir: str) -> DataFrame:
    """Read all tracking parquets from a Hive-partitioned Raw Zone subfolder.

    Expected layout:  raw_zone/{subdir}/lab_id=<lab>/<video_id>.parquet
    PySpark auto-discovers the ``lab_id`` partition column.
    """
    path = os.path.join(raw_zone, subdir)
    if not os.path.exists(path):
        print(f"  [WARN] Path not found, skipping: {path}")
        return None
    df = spark.read.option("basePath", path).parquet(path)
    # Extract video_id from the input_file_name
    df = df.withColumn(
        "_filename", F.input_file_name()
    ).withColumn(
        "video_id",
        F.regexp_extract(F.col("_filename"), r"(\d+)\.parquet", 1).cast("int"),
    ).drop("_filename")
    return df


def read_annotations(spark: SparkSession, raw_zone: str) -> DataFrame:
    """Read all annotation parquets from the Raw Zone."""
    path = os.path.join(raw_zone, "train_annotation")
    if not os.path.exists(path):
        return None
    df = spark.read.option("basePath", path).parquet(path)
    df = df.withColumn(
        "_filename", F.input_file_name()
    ).withColumn(
        "video_id",
        F.regexp_extract(F.col("_filename"), r"(\d+)\.parquet", 1).cast("int"),
    ).drop("_filename")
    return df


# =====================================================================
# Step 2b — Data cleansing & standardization (PySpark)
# =====================================================================

def clean_metadata(train_df: DataFrame) -> DataFrame:
    """Derive helper columns on the metadata DataFrame."""
    # Count mice per video (non-null strain columns)
    mouse_cols = ["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]
    train_df = train_df.withColumn(
        "n_mice",
        sum(F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0)) for c in mouse_cols),
    )
    return train_df


def normalize_tracking(tracking_df: DataFrame, metadata_df: DataFrame) -> DataFrame:
    """Join tracking with metadata and normalize pixel coords → centimeters.

    Uses SparkSQL for the join, then vectorized column arithmetic.
    """
    # Register as temp views for SparkSQL join
    tracking_df.createOrReplaceTempView("tracking")
    metadata_df.createOrReplaceTempView("metadata")

    joined = tracking_df.sparkSession.sql("""
        SELECT
            t.*,
            m.frames_per_second  AS fps,
            m.pix_per_cm_approx  AS pix_per_cm,
            m.video_width_pix    AS width_pix,
            m.video_height_pix   AS height_pix,
            m.arena_shape
        FROM tracking t
        JOIN metadata m
          ON t.video_id = m.video_id
         AND t.lab_id   = m.lab_id
    """)

    # Normalize: center on arena midpoint, convert to cm
    normalized = joined.withColumn(
        "x_cm", (F.col("x") - F.col("width_pix") / 2.0) / F.col("pix_per_cm")
    ).withColumn(
        "y_cm", (F.col("y") - F.col("height_pix") / 2.0) / F.col("pix_per_cm")
    ).withColumn(
        "x_valid", F.when(F.col("x").isNotNull(), 1).otherwise(0)
    ).withColumn(
        "y_valid", F.when(F.col("y").isNotNull(), 1).otherwise(0)
    )

    return normalized


def add_velocity_features(df: DataFrame) -> DataFrame:
    """Compute per-mouse, per-bodypart velocity using a window function.

    velocity_x = (x_cm[t] - x_cm[t-1]) * fps
    velocity_y = (y_cm[t] - y_cm[t-1]) * fps
    """
    win = (
        Window
        .partitionBy("video_id", "lab_id", "mouse_id", "bodypart")
        .orderBy("video_frame")
    )
    df = df.withColumn("prev_x", F.lag("x_cm", 1).over(win))
    df = df.withColumn("prev_y", F.lag("y_cm", 1).over(win))

    df = df.withColumn(
        "vx_cm", F.coalesce((F.col("x_cm") - F.col("prev_x")) * F.col("fps"), F.lit(0.0))
    ).withColumn(
        "vy_cm", F.coalesce((F.col("y_cm") - F.col("prev_y")) * F.col("fps"), F.lit(0.0))
    ).drop("prev_x", "prev_y")

    return df


def fill_missing(df: DataFrame) -> DataFrame:
    """Fill NaN coordinates with 0 (after marking validity masks above)."""
    df = df.fillna({"x_cm": 0.0, "y_cm": 0.0, "vx_cm": 0.0, "vy_cm": 0.0})
    return df


# =====================================================================
# Step 2c — Annotation enrichment (SparkSQL)
# =====================================================================

def enrich_annotations(anno_df: DataFrame) -> DataFrame:
    """Add derived columns to annotation events using SparkSQL."""
    anno_df.createOrReplaceTempView("annotations")

    enriched = anno_df.sparkSession.sql("""
        SELECT
            *,
            (stop_frame - start_frame) AS duration_frames,
            CASE
                WHEN action IN ({self_list})
                THEN 'self'
                ELSE 'interaction'
            END AS action_type
        FROM annotations
    """.format(
        self_list=", ".join(f"'{a}'" for a in SELF_ACTIONS)
    ))
    return enriched


def compute_class_distribution(anno_df: DataFrame) -> DataFrame:
    """Aggregate class distribution stats using SparkSQL."""
    anno_df.createOrReplaceTempView("anno_enriched")

    stats = anno_df.sparkSession.sql("""
        SELECT
            lab_id,
            action,
            action_type,
            COUNT(*)                       AS event_count,
            SUM(duration_frames)           AS total_frames,
            ROUND(AVG(duration_frames), 1) AS avg_duration,
            PERCENTILE_APPROX(duration_frames, 0.5) AS median_duration
        FROM anno_enriched
        GROUP BY lab_id, action, action_type
        ORDER BY lab_id, event_count DESC
    """)
    return stats


# =====================================================================
# Step 2d — Write to Curated Zone (partitioned Parquet)
# =====================================================================

def write_curated(df: DataFrame, name: str, curated_zone: str,
                  partition_cols: list[str] | None = None) -> None:
    """Write a DataFrame to the Curated Zone in Parquet format."""
    out_path = os.path.join(curated_zone, name)
    writer = df.write.mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(out_path)
    print(f"  ✓ Wrote {name} → {out_path}")


# =====================================================================
# Main ETL orchestrator
# =====================================================================

def run_etl() -> None:
    """Full ETL pipeline: Raw Zone → Curated Zone."""
    print("=" * 60)
    print("  AWS Glue ETL Simulation — MABe Mouse Behavior")
    print("=" * 60)

    raw_zone = LOCAL_RAW_ZONE
    curated_zone = LOCAL_CURATED_ZONE
    os.makedirs(curated_zone, exist_ok=True)

    spark = create_spark_session()
    print(f"\n[1/6] Reading metadata from Raw Zone ...")
    train_meta, test_meta = read_metadata(spark, raw_zone)
    train_meta = clean_metadata(train_meta)
    test_meta = clean_metadata(test_meta)

    # Write curated metadata
    write_curated(train_meta, "metadata_train", curated_zone)
    write_curated(test_meta, "metadata_test", curated_zone)
    print(f"       Train videos: {train_meta.count()}, Test videos: {test_meta.count()}")

    print(f"\n[2/6] Reading train tracking from Raw Zone ...")
    train_tracking = read_tracking(spark, raw_zone, "train_tracking")
    if train_tracking is not None:
        row_count = train_tracking.count()
        print(f"       Tracking rows: {row_count:,}")

        print(f"\n[3/6] Normalizing coordinates & adding velocity features ...")
        curated_tracking = normalize_tracking(train_tracking, train_meta)
        curated_tracking = add_velocity_features(curated_tracking)
        curated_tracking = fill_missing(curated_tracking)

        # Write partitioned by lab_id
        write_curated(curated_tracking, "tracking_curated", curated_zone,
                      partition_cols=["lab_id"])
    else:
        print("       [SKIP] No train tracking data found.")

    print(f"\n[4/6] Reading annotations from Raw Zone ...")
    annotations = read_annotations(spark, raw_zone)
    if annotations is not None:
        print(f"       Annotation events: {annotations.count():,}")

        print(f"\n[5/6] Enriching annotations (SparkSQL) ...")
        enriched = enrich_annotations(annotations)
        write_curated(enriched, "annotations_enriched", curated_zone,
                      partition_cols=["lab_id"])

        print(f"\n[6/6] Computing class distribution stats (SparkSQL) ...")
        class_stats = compute_class_distribution(enriched)
        write_curated(class_stats, "class_distribution_stats", curated_zone)
        class_stats.show(20, truncate=False)
    else:
        print("       [SKIP] No annotation data found.")

    spark.stop()
    print("\n" + "=" * 60)
    print("  ETL COMPLETE — Curated Zone ready for Athena / EDA")
    print("=" * 60)


if __name__ == "__main__":
    run_etl()
