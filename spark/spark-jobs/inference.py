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
  ``timestamp`` descending.  GridFS + ``model_registry`` are canonical (M6):
  the CI MLflow file store is ephemeral, so MLflow is only tried when the
  registry has no document (or the document carries no ``gridfs_file_id``).
- Which document is scored is decided by ``model_selection.select_entry``:
  ``stage="production"`` when a human promoted one, otherwise the newest
  ``staging`` document with a logged warning (never a silent promotion).
- We load *both* models (temp + rain) so we can store both predictions in
  a single document per city per inference run.
- Shadow A/B (M5): when a production model is selected and a newer ``staging``
  document exists for the same prefix, the shadow model scores the same cached
  feature frame and ONE idempotent doc per ``(production_id, shadow_id)`` goes
  into ``model_shadow``.  Pieces are bounded by construction: O(#model-pairs),
  never O(#predictions) — no prediction rows are written by the A/B path, and
  a shadow failure logs and is skipped rather than failing the run.
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
from datetime import datetime, timezone

import model_selection
from gridfs import GridFS
from mongo_retry import run_with_mongo_retries
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

#: Shadow A/B collection (M5): one doc per (production model, shadow model) pair.
SHADOW_COLLECTION = "model_shadow"

#: Rain probability treated as "rain" when counting shadow/production disagreement.
SHADOW_RAIN_THRESHOLD = 0.5


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


def load_model_entry(fs, entry) -> PipelineModel:
    """Load the artifact of one registry *entry* from GridFS (canonical).

    GridFS is the canonical artifact store (M6) and the only one that carries a
    deterministic ``_id`` for a registry document, so a document with a
    ``gridfs_file_id`` is *always* served from GridFS. Used for the production
    model and for shadow A/B challengers alike.
    """
    file_id = entry["gridfs_file_id"]
    local_dir = _download_and_unzip(fs, file_id, TMP_DIR)
    try:
        model = PipelineModel.load(local_dir)
        logger.info(
            "Loaded model '%s' (version %s).", entry.get("model_name"), entry.get("version")
        )
        return model
    finally:
        # Clean up extracted directory; keep nothing on disk
        shutil.rmtree(local_dir, ignore_errors=True)


def _load_from_mlflow(model_name_prefix: str):
    """Try the ephemeral MLflow registry; return ``(model, derived_meta)`` or ``(None, None)``.

    Only reached when ``model_registry`` cannot supply the artifact. The
    returned metadata is deliberately minimal and labelled ``MLflow-*`` so a
    row can never claim registry lineage it does not have.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        return None, None

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
                    logger.info("Attempting to load model '%s' from MLflow Registry...", model_uri)
                    model = mlflow.spark.load_model(model_uri)
                    logger.info("Successfully loaded model from MLflow Registry: %s", model_uri)
                    return model, {"model_name": cand, "version": f"MLflow-{stage}"}
                except Exception:
                    continue
    except Exception as e:
        logger.info("MLflow model load skipped/failed (%s).", e)

    return None, None


def load_latest_model(db, fs, model_name_prefix: str, spark):
    """Fetch the selected (production, else staging) model and its registry metadata.

    Order of resolution is frozen by Batch 5:

    1. the registry document comes from ``model_selection.select_entry`` (so
       ``stage="production"`` wins; staging is only a flagged fallback);
    2. the **artifact** is loaded from **GridFS first** whenever the document
       carries a ``gridfs_file_id``;
    3. MLflow (``models:/{cand}/{stage}``) is only attempted when there is no
       registry document at all, or the document has no ``gridfs_file_id`` —
       and even then the registry's metadata wins, because the MLflow file store
       is ephemeral and carries no interval/lineage.

    Returns ``(PipelineModel, metadata_dict)`` or ``(None, None)``.
    """
    entry, is_fallback = model_selection.select_entry(db, model_name_prefix)
    if is_fallback:
        logger.warning(
            "No production model for prefix '%s'; falling back to staging model '%s' "
            "(promote one with scripts/promote_model.py).",
            model_name_prefix,
            entry.get("model_name"),
        )

    if entry is not None and entry.get("gridfs_file_id"):
        return load_model_entry(fs, entry), entry

    model, derived_meta = _load_from_mlflow(model_name_prefix)
    if model is not None:
        # The registry document, when it exists, stays the metadata source even
        # though MLflow served the artifact.
        return model, entry if entry is not None else derived_meta

    if entry is None:
        logger.warning("No model found for prefix '%s'. Skipping.", model_name_prefix)
        return None, None

    logger.warning(
        "Registry entry '%s' (%s) has no gridfs_file_id and MLflow served no artifact; "
        "skipping prefix '%s'.",
        entry.get("model_name"),
        entry.get("_id"),
        model_name_prefix,
    )
    return None, None


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

    # Materialise once: every horizon's transform and the final write reuse this
    # frame, so caching removes ~10 full re-scans of weather_features per run.
    df_latest = df_latest.cache()

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


def utc_epoch_seconds(column: str = "timestamp"):
    """Epoch seconds for a TIMESTAMP *column*, pinned to UTC.

    Spark renders and parses TIMESTAMP values with the *session* time zone, so a
    naive epoch expression can mint a different ``_id`` for the same prediction
    after a TZ change (the prediction key would drift and re-insert rows).
    ``create_spark_session`` pins the session to UTC; this expression makes the
    value independent of that setting as well by parsing the timestamp in the
    explicit ``UTC`` zone, which a unit test locks against a non-UTC session TZ.
    """
    return F.unix_timestamp(F.to_utc_timestamp(F.col(column), F.lit("UTC")))


def build_output_df(
    pred_df, temp_meta: dict | None = None, rain_meta: dict | None = None, horizon=None
):
    """Add the additive interval columns and project the final write schema.

    ``temp_lower`` / ``temp_upper`` / ``interval_level`` come from the loaded
    temperature model's ``interval`` registry block; all three are null when the
    model carries no interval (legacy document) or no temperature model loaded.
    ``horizon`` drives ``horizon_hours`` and the ``_id`` suffix; it defaults to
    the first configured horizon for callers that build a single horizon.
    """
    if horizon is None:
        horizon = FEATURES_CONFIG["target_horizons"][0]
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
            utc_epoch_seconds("timestamp").cast("string"),
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
    horizon: int,
):
    """Apply both models to *df_features* for one *horizon* and return that
    horizon's output rows (not yet written). The caller unions every horizon's
    rows into a single upsert write."""

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

    return build_output_df(pred_df, temp_meta, rain_meta, horizon)


def write_predictions(output, mongo_url: str):
    """Upsert the unioned prediction rows into ``weather_predictions``."""

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
# Shadow A/B
# ---------------------------------------------------------------------------


def _model_predictions(model, df, alias: str):
    """Score *df* with *model* and keep ``city``/``timestamp``/``alias``.

    Scratch columns (``features_raw``/``features``) are dropped so a second
    pipeline can run on the same frame — see ``_pipeline_scratch_columns``.
    """
    out = model.transform(df).withColumnRenamed("prediction", alias)
    scratch = _pipeline_scratch_columns(model)
    if scratch:
        out = out.drop(*scratch)
    return out.select("city", "timestamp", F.col(alias).cast("double").alias(alias))


def _as_utc_datetime(epoch_seconds):
    """``datetime`` (UTC-aware) from an epoch integer; None stays None.

    Collecting a TIMESTAMP column into Python yields a *naive* wall clock in the
    session time zone, which pymongo would then store as if it were UTC. Going
    through the pinned UTC epoch keeps the stored value honest.
    """
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc)


def _as_float(value):
    """Mongo-safe float (NaN/inf and Spark nulls become None)."""
    if value is None:
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def shadow_comparison(df_features, production_model, shadow_model, target: str) -> dict:
    """Compare production and shadow predictions on the SAME feature frame.

    One aggregation over the cached frame; no prediction rows are written. The
    join is on ``(city, timestamp)`` so a model that silently drops rows cannot
    shift the deltas, and null predictions are excluded from both sides.
    """
    production = _model_predictions(production_model, df_features, "pred_production")
    shadow = _model_predictions(shadow_model, df_features, "pred_shadow")

    joined = production.join(shadow, on=["city", "timestamp"], how="inner").filter(
        F.col("pred_production").isNotNull() & F.col("pred_shadow").isNotNull()
    )

    delta = F.abs(F.col("pred_shadow") - F.col("pred_production"))
    row = joined.agg(
        F.count(F.lit(1)).alias("n_samples"),
        F.max(utc_epoch_seconds("timestamp")).alias("last_source_epoch"),
        F.mean(delta).alias("delta_mean"),
        F.max(delta).alias("delta_max"),
        F.mean(
            F.when(
                (F.col("pred_production") >= SHADOW_RAIN_THRESHOLD)
                != (F.col("pred_shadow") >= SHADOW_RAIN_THRESHOLD),
                1.0,
            ).otherwise(0.0)
        ).alias("disagreement_rate"),
    ).first()

    metrics = {
        "n_samples": int(row["n_samples"] or 0),
        "last_source_timestamp": _as_utc_datetime(row["last_source_epoch"]),
    }
    if target == "rain":
        metrics["rain_prob_delta_mean"] = _as_float(row["delta_mean"])
        metrics["rain_disagreement_rate"] = _as_float(row["disagreement_rate"])
    else:
        metrics["temp_abs_delta_mean"] = _as_float(row["delta_mean"])
        metrics["temp_abs_delta_max"] = _as_float(row["delta_max"])
    return metrics


def shadow_doc_id(production_id, shadow_id) -> str:
    """Stable ``model_shadow`` ``_id`` — one doc per (production, shadow) pair."""
    return f"{production_id}_{shadow_id}"


def build_shadow_doc(
    production_entry: dict,
    shadow_entry_doc: dict,
    target: str,
    horizon: int,
    metrics: dict,
    updated_at: datetime | None = None,
) -> dict:
    """Registry-shape document for ``model_shadow`` (pure: no Spark, no Mongo)."""
    doc = {
        "_id": shadow_doc_id(production_entry.get("_id"), shadow_entry_doc.get("_id")),
        "target": target,
        "horizon_hours": horizon,
        "production_model_name": production_entry.get("model_name"),
        "production_model_version": production_entry.get("version"),
        "shadow_model_name": shadow_entry_doc.get("model_name"),
        "shadow_model_version": shadow_entry_doc.get("version"),
        "last_source_timestamp": metrics.get("last_source_timestamp"),
        "n_samples": int(metrics.get("n_samples") or 0),
        "updated_at": updated_at or datetime.now(timezone.utc),
    }
    if target == "rain":
        doc["rain_prob_delta_mean"] = metrics.get("rain_prob_delta_mean")
        doc["rain_disagreement_rate"] = metrics.get("rain_disagreement_rate")
    else:
        doc["temp_abs_delta_mean"] = metrics.get("temp_abs_delta_mean")
        doc["temp_abs_delta_max"] = metrics.get("temp_abs_delta_max")
    return doc


def write_shadow(db, doc: dict) -> None:
    """Idempotent write: ``replace_one(_id, upsert=True)`` — a re-run never duplicates."""
    db[SHADOW_COLLECTION].replace_one({"_id": doc["_id"]}, doc, upsert=True)


def run_shadow_ab(
    db,
    fs,
    prefix: str,
    production_model,
    production_entry: dict,
    df_features,
    horizon: int,
    target: str,
) -> dict | None:
    """Score the shadow challenger and upsert its A/B doc; never raises.

    Skipped (returns None) when there is no production model, no staged
    challenger, no comparable rows, or the challenger cannot be loaded/run: a
    shadow fault must never fail inference.
    """
    if production_model is None or not production_entry:
        return None
    if production_entry.get("stage") != model_selection.STAGE_PRODUCTION:
        return None

    try:
        challenger = model_selection.shadow_entry(db, prefix, production_entry.get("_id"))
        if challenger is None or not challenger.get("gridfs_file_id"):
            return None

        shadow_model = load_model_entry(fs, challenger)
        metrics = shadow_comparison(df_features, production_model, shadow_model, target)
        if metrics["n_samples"] == 0:
            logger.info("Shadow A/B '%s' skipped: no comparable rows for %sh.", prefix, horizon)
            return None

        doc = build_shadow_doc(production_entry, challenger, target, horizon, metrics)
        run_with_mongo_retries(
            lambda: write_shadow(db, doc),
            label=f"write shadow A/B {doc['_id']}",
        )
        logger.info(
            "Shadow A/B '%s': production=%s shadow=%s n=%d",
            doc["_id"],
            doc["production_model_version"],
            doc["shadow_model_version"],
            doc["n_samples"],
        )
        return doc
    except Exception as exc:
        logger.warning("Shadow A/B for '%s' skipped after %s: %s", prefix, type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    mongo_url = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
    if not mongo_url:
        raise RuntimeError("MONGO_URI environment variable is not set.")

    spark = create_spark_session("WeatherInference")
    df_features = None

    try:
        client = MongoClient(mongo_url)
        db = client["weather_db"]
        fs = GridFS(db)

        horizons = FEATURES_CONFIG["target_horizons"]

        df_features = run_with_mongo_retries(
            lambda: load_latest_features(spark, mongo_url),
            label="load latest features",
        )

        if df_features.count() == 0:
            logger.warning("No feature rows found in weather_features. Nothing to predict.")
            return

        output = None
        shadow_jobs: list[tuple] = []
        for horizon in horizons:
            temp_prefix = f"temp_prediction_{horizon}h"
            rain_prefix = f"rain_prediction_{horizon}h"

            temp_model, temp_meta = run_with_mongo_retries(
                lambda: load_latest_model(db, fs, temp_prefix, spark),
                label=f"load {temp_prefix} model",
            )
            rain_model, rain_meta = run_with_mongo_retries(
                lambda: load_latest_model(db, fs, rain_prefix, spark),
                label=f"load {rain_prefix} model",
            )

            if temp_model is None and rain_model is None:
                # A missing horizon is skipped, never fatal: the remaining
                # horizons still produce predictions.
                logger.warning(
                    "No trained models found for horizon %sh. Skipping this horizon.",
                    horizon,
                )
                continue

            horizon_output = run_inference(
                spark,
                temp_model,
                rain_model,
                df_features,
                mongo_url,
                temp_meta or {},
                rain_meta or {},
                horizon,
            )
            output = horizon_output if output is None else output.unionByName(horizon_output)

            # Shadow A/B (M5) candidates, deferred until after the prediction
            # write so the A/B can never delay the critical path. It runs only
            # when a real production model was selected — never on the staging
            # fallback.
            for target, prefix, prod_model, prod_meta in (
                ("temperature", temp_prefix, temp_model, temp_meta),
                ("rain", rain_prefix, rain_model, rain_meta),
            ):
                if not prod_meta or prod_meta.get("stage") != model_selection.STAGE_PRODUCTION:
                    continue
                if prod_model is None:
                    continue
                shadow_jobs.append((target, prefix, prod_model, prod_meta, horizon))

        if output is None:
            logger.warning("No horizon produced predictions. Run ml_training.py first.")
            return

        # Retrying this write cannot duplicate rows: it is an idempotent upsert
        # keyed on _id (operationType=replace + upsertDocument=true), so a re-run
        # replaces the same documents instead of appending.
        run_with_mongo_retries(
            lambda: write_predictions(output, mongo_url),
            label="write predictions",
        )

        # Shadow A/B (M5): bounded to one doc per (production, shadow) pair,
        # scored on the SAME cached frame, and never fatal — run_shadow_ab logs
        # and skips on any fault.
        for target, prefix, prod_model, prod_meta, horizon in shadow_jobs:
            run_shadow_ab(
                db,
                fs,
                prefix,
                prod_model,
                prod_meta,
                df_features,
                horizon,
                target,
            )

    except Exception as exc:
        logger.error("Inference job failed: %s", exc)
        raise
    finally:
        if df_features is not None:
            df_features.unpersist()
        client.close()
        spark.stop()


if __name__ == "__main__":
    main()
