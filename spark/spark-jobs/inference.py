"""
inference.py
============
Spark batch inference job.

Loads the most recent temperature-prediction and rain-prediction models
from GridFS (as saved by ml_training.py), runs predictions on the latest
feature row per city, and appends the results to the weather_predictions
collection in MongoDB.

Design decisions
----------------
- Models are stored as zipped Spark PipelineModel directories in GridFS.
  The entry point metadata lives in ``model_registry``, ordered by
  ``timestamp`` descending.
- We load *both* models (temp + rain) so we can store both predictions in
  a single document per city per inference run.
- The job is idempotent at the city level: we upsert on (city, source_timestamp)
  rather than blindly appending, so re-runs after a failure don't create
  duplicates.
- Runs without a Spark master when SPARK_MASTER env-var is not set (useful
  for local testing with master="local[*]").
"""

import os
import sys
import shutil
import zipfile
import logging

from pymongo import MongoClient
from gridfs import GridFS

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

sys.path.append("/opt/config")
from spark_config import create_spark_session, FEATURES_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TMP_DIR = "/opt/spark-tmp/inference"


def _latest_model_entry(db, model_name_prefix: str) -> dict | None:
    """Return the most recent model_registry document whose name starts with
    *model_name_prefix*, or None if no model has been trained yet."""
    return db["model_registry"].find_one(
        {"model_name": {"$regex": f"^{model_name_prefix}"}},
        sort=[("timestamp", -1)],
    )


def _download_and_unzip(fs: GridFS, file_id, dest_dir: str) -> str:
    """Download a GridFS file, unzip it into *dest_dir*, and return the
    path to the extracted directory."""
    zip_path = os.path.join(dest_dir, f"{file_id}.zip")
    model_path = os.path.join(dest_dir, str(file_id))

    os.makedirs(dest_dir, exist_ok=True)

    logger.info("Downloading model zip from GridFS (file_id=%s)…", file_id)
    with open(zip_path, "wb") as fh:
        fh.write(fs.get(file_id).read())

    logger.info("Extracting model to %s…", model_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_path)

    os.remove(zip_path)
    return model_path


def load_latest_model(db, fs, model_name_prefix: str, spark):
    """Fetch the latest matching model from GridFS and load it as a
    PipelineModel. Returns (PipelineModel, metadata_dict) or (None, None)."""
    entry = _latest_model_entry(db, model_name_prefix)
    if entry is None:
        logger.warning("No model found for prefix '%s'. Skipping.", model_name_prefix)
        return None, None

    file_id = entry["gridfs_file_id"]
    local_dir = _download_and_unzip(fs, file_id, TMP_DIR)

    try:
        model = PipelineModel.load(local_dir)
        logger.info(
            "Loaded model '%s' (version %s).", entry["model_name"], entry.get("version")
        )
        return model, entry
    finally:
        # Clean up extracted directory; keep nothing on disk
        shutil.rmtree(local_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_latest_features(spark, mongo_url: str):
    """Read weather_features from Mongo and return only the most-recent row
    per city (using MAX(timestamp))."""
    df = (
        spark.read.format("mongodb")
        .option("connection.uri", mongo_url)
        .option("database", "weather_db")
        .option("collection", "weather_features")
        .load()
    )

    if "_id" in df.columns:
        df = df.drop("_id")

    # Keep latest record per city
    latest = df.groupBy("city").agg(F.max("timestamp").alias("timestamp"))
    df_latest = latest.join(df, on=["city", "timestamp"], how="inner")

    city_count = df_latest.count()
    logger.info("Loaded latest features for %d cities.", city_count)
    return df_latest


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def run_inference(
    spark,
    temp_model,
    rain_model,
    df_features,
    mongo_url: str,
    temp_meta: dict,
    rain_meta: dict,
):
    """Apply both models to *df_features* and upsert predictions into
    weather_predictions."""

    pred_df = df_features

    if temp_model is not None:
        pred_df = temp_model.transform(pred_df).withColumnRenamed(
            "prediction", "predicted_temperature"
        )
    else:
        pred_df = pred_df.withColumn(
            "predicted_temperature", F.lit(None).cast("double")
        )

    if rain_model is not None:
        pred_df = rain_model.transform(pred_df).withColumnRenamed(
            "prediction", "predicted_rain"
        )
    else:
        pred_df = pred_df.withColumn("predicted_rain", F.lit(None).cast("double"))

    horizon = FEATURES_CONFIG["target_horizon"]

    output = pred_df.select(
        F.col("city"),
        F.col("timestamp").alias("source_timestamp"),
        F.current_timestamp().alias("prediction_timestamp"),
        F.col("predicted_temperature"),
        F.col("predicted_rain"),
        # Include the observed temperature so callers can compute error on-the-fly
        F.col("temperature").alias("observed_temperature"),
        F.lit(horizon).alias("horizon_hours"),
        F.lit(temp_meta["model_name"] if temp_meta else "unknown").alias(
            "temp_model_name"
        ),
        F.lit(temp_meta["version"] if temp_meta else "unknown").alias(
            "temp_model_version"
        ),
        F.lit(rain_meta["model_name"] if rain_meta else "unknown").alias(
            "rain_model_name"
        ),
        F.lit(rain_meta["version"] if rain_meta else "unknown").alias(
            "rain_model_version"
        ),
    )

    row_count = output.count()
    logger.info("Writing %d prediction rows to weather_predictions…", row_count)

    output.write.format("mongodb").option("connection.uri", mongo_url).option(
        "database", "weather_db"
    ).option("collection", "weather_predictions").mode("append").save()

    logger.info("Predictions saved successfully.")
    output.show(truncate=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    mongo_url = os.getenv("MONGO_URL")
    if not mongo_url:
        raise RuntimeError("MONGO_URL environment variable is not set.")

    spark = create_spark_session("WeatherInference")

    try:
        client = MongoClient(mongo_url)
        db = client["weather_db"]
        fs = GridFS(db)

        horizon = FEATURES_CONFIG["target_horizon"]
        temp_prefix = f"temp_prediction_{horizon}h"
        rain_prefix = f"rain_prediction_{horizon}h"

        temp_model, temp_meta = load_latest_model(db, fs, temp_prefix, spark)
        rain_model, rain_meta = load_latest_model(db, fs, rain_prefix, spark)

        if temp_model is None and rain_model is None:
            logger.warning(
                "No trained models found. Run ml_training.py at least once before inference."
            )
            return

        df_features = load_latest_features(spark, mongo_url)

        if df_features.count() == 0:
            logger.warning(
                "No feature rows found in weather_features. Nothing to predict."
            )
            return

        run_inference(
            spark,
            temp_model,
            rain_model,
            df_features,
            mongo_url,
            temp_meta or {},
            rain_meta or {},
        )

    except Exception as exc:
        logger.error("Inference job failed: %s", exc)
        raise
    finally:
        client.close()
        spark.stop()


if __name__ == "__main__":
    main()
