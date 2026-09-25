"""
inference.py
============
Spark batch inference job.

Loads the most recent temperature-prediction and rain-prediction models
from GridFS (as saved by ml_training.py), runs predictions on the latest
feature row per city, and upserts the results into the weather_predictions
collection in MongoDB.

Design decisions
----------------
- Models are stored as zipped Spark PipelineModel directories in GridFS.
  The entry point metadata lives in ``model_registry``, ordered by
  ``timestamp`` descending.
- We load *both* models (temp + rain) so we can store both predictions in
  a single document per city per inference run.
- The job is idempotent: every prediction row carries a stable ``_id`` derived
  from (city, source_timestamp, horizon), and the write uses the connector's
  replace (upsert) operation, so re-runs after a failure replace the same
  document instead of appending duplicates.
- Runs without a Spark master when SPARK_MASTER env-var is not set (useful
  for local testing with master="local[*]").
"""

import logging
import os
import shutil
import sys
import zipfile

from gridfs import GridFS
from pymongo import MongoClient
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

sys.path.append("/opt/config")
from spark_config import FEATURES_CONFIG, create_spark_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TMP_DIR = os.path.join(os.getenv("SPARK_TMP_DIR", "/opt/spark-tmp"), "inference")


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
    """Fetch the latest matching model from MLflow Registry or GridFS fallback and load it as a
    PipelineModel. Returns (PipelineModel, metadata_dict) or (None, None)."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        try:
            import mlflow.spark

            mlflow.set_tracking_uri(mlflow_uri)
            candidate_names = [model_name_prefix]
            if "temp_prediction" in model_name_prefix:
                candidate_names.append(
                    model_name_prefix.replace("temp_prediction", "weather_temperature")
                )
            elif "rain_prediction" in model_name_prefix:
                candidate_names.append(model_name_prefix.replace("rain_prediction", "weather_rain"))

            for cand in candidate_names:
                for stage in ["Production", "latest", "None"]:
                    try:
                        model_uri = f"models:/{cand}/{stage}"
                        logger.info(
                            "Attempting to load model '%s' from MLflow Registry...", model_uri
                        )
                        model = mlflow.spark.load_model(model_uri)
                        meta = {"model_name": cand, "version": f"MLflow-{stage}"}
                        logger.info("Successfully loaded model from MLflow Registry: %s", model_uri)
                        return model, meta
                    except Exception:
                        continue
        except Exception as e:
            logger.info("MLflow model load skipped/failed (%s), falling back to GridFS...", e)

    entry = _latest_model_entry(db, model_name_prefix)
    if entry is None:
        logger.warning("No model found for prefix '%s'. Skipping.", model_name_prefix)
        return None, None

    file_id = entry["gridfs_file_id"]
    local_dir = _download_and_unzip(fs, file_id, TMP_DIR)

    try:
        model = PipelineModel.load(local_dir)
        logger.info("Loaded model '%s' (version %s).", entry["model_name"], entry.get("version"))
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


def _pipeline_scratch_columns(model) -> list[str]:
    """Intermediate columns a pipeline adds besides ``prediction``.

    Training names the assembler output ``features_raw`` and the scaler
    output ``features`` (see ``prepare_features_for_ml``). The second model's
    pipeline writes the same names, so Spark raises
    ``IllegalArgumentException: Output column features_raw already exists``
    unless the first model's scratch columns are dropped in between.
    """
    scratch = []
    for stage in getattr(model, "stages", []):
        if stage.hasParam("outputCol"):
            column = stage.getOutputCol()
            if column != "prediction":
                scratch.append(column)
    return scratch


def output_schema_columns() -> list[str]:
    """Exact write-column names, in write order, for ``weather_predictions``.

    Pure and importable without a Spark session or a database connection; the
    CI output-schema guard imports this and asserts the contract verbatim.
    """
    return [
        "_id",
        "city",
        "source_timestamp",
        "prediction_timestamp",
        "predicted_temperature",
        "predicted_rain",
        "observed_temperature",
        "horizon_hours",
        "temp_model_name",
        "temp_model_version",
        "rain_model_name",
        "rain_model_version",
        "temp_lower",
        "temp_upper",
        "interval_level",
    ]


def build_output_df(pred_df, temp_meta: dict | None = None, rain_meta: dict | None = None):
    """Add the additive interval columns and project the final write schema.

    ``temp_lower`` / ``temp_upper`` / ``interval_level`` come from the loaded
    temperature model's ``interval`` registry block; all three are null when the
    model carries no interval (legacy document) or no temperature model loaded.
    """
    horizon = FEATURES_CONFIG["target_horizon"]
    temp_meta = temp_meta or {}
    rain_meta = rain_meta or {}

    interval = temp_meta.get("interval") or {}
    lower = interval.get("lower_offset")
    upper = interval.get("upper_offset")
    level = interval.get("level")
    if lower is not None and upper is not None and level is not None:
        pred_df = pred_df.withColumn(
            "temp_lower", F.col("predicted_temperature") + F.lit(float(lower))
        )
        pred_df = pred_df.withColumn(
            "temp_upper", F.col("predicted_temperature") + F.lit(float(upper))
        )
        pred_df = pred_df.withColumn("interval_level", F.lit(float(level)))
    else:
        pred_df = pred_df.withColumn("temp_lower", F.lit(None).cast("double"))
        pred_df = pred_df.withColumn("temp_upper", F.lit(None).cast("double"))
        pred_df = pred_df.withColumn("interval_level", F.lit(None).cast("double"))

    expressions = {
        "_id": F.concat_ws(
            "_",
            F.col("city"),
            F.unix_timestamp("timestamp").cast("string"),
            F.lit(f"{horizon}h"),
        ),
        "city": F.col("city"),
        "source_timestamp": F.col("timestamp"),
        "prediction_timestamp": F.current_timestamp(),
        "predicted_temperature": F.col("predicted_temperature"),
        "temp_lower": F.col("temp_lower"),
        "temp_upper": F.col("temp_upper"),
        "interval_level": F.col("interval_level"),
        "predicted_rain": F.col("predicted_rain"),
        "observed_temperature": F.col("temperature"),
        "horizon_hours": F.lit(horizon),
        "temp_model_name": F.lit(temp_meta.get("model_name", "unknown")),
        "temp_model_version": F.lit(temp_meta.get("version", "unknown")),
        "rain_model_name": F.lit(rain_meta.get("model_name", "unknown")),
        "rain_model_version": F.lit(rain_meta.get("version", "unknown")),
    }

    return pred_df.select(*[expressions[name].alias(name) for name in output_schema_columns()])


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
        scratch = _pipeline_scratch_columns(temp_model)
        if scratch:
            pred_df = pred_df.drop(*scratch)
    else:
        pred_df = pred_df.withColumn("predicted_temperature", F.lit(None).cast("double"))

    if rain_model is not None:
        pred_df = rain_model.transform(pred_df).withColumnRenamed("prediction", "predicted_rain")
        scratch = _pipeline_scratch_columns(rain_model)
        if scratch:
            pred_df = pred_df.drop(*scratch)
    else:
        pred_df = pred_df.withColumn("predicted_rain", F.lit(None).cast("double"))

    if temp_model is not None and not (temp_meta or {}).get("interval"):
        logger.warning(
            "Loaded temperature model has no 'interval' block; "
            "temp_lower/temp_upper/interval_level will be null."
        )

    output = build_output_df(pred_df, temp_meta, rain_meta)

    row_count = output.count()
    logger.info("Writing %d prediction rows to weather_predictions…", row_count)

    output.write.format("mongodb").option("connection.uri", mongo_url).option(
        "database", "weather_db"
    ).option("collection", "weather_predictions").option(
        # Upsert on _id. The 10.x connector calls this operationType/upsertDocument
        # (both already default to "replace"/true); `replaceDocument` is the old
        # 3.x option name. `mode("append")` here only means "don't drop the
        # collection" — operationType=replace makes each _id replace (upsert) its row.
        "operationType",
        "replace",
    ).option(
        "upsertDocument", "true"
    ).mode(
        "append"
    ).save()

    logger.info("Predictions saved successfully.")
    output.show(truncate=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    mongo_url = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
    if not mongo_url:
        raise RuntimeError("MONGO_URI environment variable is not set.")

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
            logger.warning("No feature rows found in weather_features. Nothing to predict.")
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
