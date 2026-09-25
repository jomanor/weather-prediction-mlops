import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, ShortType

sys.path.append("/opt/config")
import json
import math
import os
import shutil
from datetime import datetime

import mlflow
import mlflow.spark
from gridfs import GridFS
from pymongo import MongoClient
from spark_config import FEATURES_CONFIG, ML_CONFIG, create_spark_session


def load_features(spark):
    mongo_url = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")

    df = (
        spark.read.format("mongodb")
        .option("connection.uri", mongo_url)
        .option("database", "weather_db")
        .option("collection", "weather_features")
        .load()
    )

    df = df.drop("_id")

    # A blanket dropna() here deleted every row. weather_features is a union of
    # two generations of documents, and a column that only one of them wrote is
    # null in all the others, so no row is complete across every column. Keep
    # what training cannot invent -- the observation -- and let the per-horizon
    # loop drop its own label columns, then let the assembler skip rows with gaps
    # in individual features. Targets are deliberately NOT dropped here: with
    # several horizons, requiring every ``target_*`` column non-null would throw
    # away the newest ~24h of observations per city.
    df_clean = df.dropna(subset=["city", "timestamp", "temperature"])

    # An overlapping scheduled run can write the same city+hour twice; that is
    # one observation, so keep a single sample per hour rather than one per time
    # the job happened to run.
    return df_clean.dropDuplicates(["city", "timestamp"])


def prepare_features_for_ml(df, target_col, horizon=1):
    target_temp_col = f"target_temp_{horizon}h"
    target_rain_col = f"target_will_rain_{horizon}h"

    exclude_cols = [
        "city",
        "timestamp",
        "weather_main",
        "weather_description",
        target_col,
        target_temp_col,
        target_rain_col,
    ]
    # Every horizon's target columns are written by ``create_target_variable``
    # now, not just this one. A future-horizon label (e.g. ``target_temp_3h``)
    # is numeric, ~always present, and would otherwise survive the 50%-presence
    # filter and leak the actual future temperature into the features — and the
    # newest feature row has all of them null, so the assembler would drop it.
    # Exclude all ``target_*`` regardless of horizon.
    exclude_cols += [c for c in df.columns if c.startswith("target_")]

    # A field the connector could not infer (absent, or null in every sampled
    # document) arrives as `void`, and VectorAssembler rejects that type
    # outright. Text, arrays and booleans are equally unusable as features, so
    # keep only the numeric columns the assembler can actually accept.
    numeric_types = (DoubleType, FloatType, IntegerType, LongType, ShortType)
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and isinstance(df.schema[col].dataType, numeric_types)
    ]

    # A column that is empty in most rows costs every row it touches, because
    # the assembler discards a row as soon as one of its inputs is null.
    # cloud_coverage, for instance, is written only by the older generation of
    # feature documents and is null in 93% of them, so keeping it would throw
    # away almost the entire history to gain a feature that carries nothing.
    total = df.count()
    present = df.select([F.count(F.col(c)).alias(c) for c in feature_cols]).first().asDict()
    feature_cols = [c for c in feature_cols if present[c] >= total * 0.5]
    if not feature_cols:
        raise ValueError("No usable feature columns: every numeric column is mostly null.")

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip"
    )

    scaler = StandardScaler(inputCol="features_raw", outputCol="features")

    return assembler, scaler, feature_cols


# ---------------------------------------------------------------------------
# Temporal split + honest-metric helpers
# ---------------------------------------------------------------------------


def _iso(value):
    """Convert a Spark timestamp / Python datetime to an ISO-8601 string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def temporal_split(df):
    """Split *df* chronologically by ``timestamp`` at the cumulative train/val
    ratios from ``ML_CONFIG["data_split"]``.

    The cut is made on the **timestamp value**, not the row index: the ordered
    set of distinct timestamps is collected once (tiny for ~180 days of hourly
    data) and two boundary timestamps ``b1``/``b2`` are chosen at the cumulative
    ratio positions. Rows are then partitioned with ``ts < b1`` / ``b1 <= ts <
    b2`` / ``ts >= b2``, so a given timestamp never appears in more than one
    frame and no single-partition sort is involved. The recorded boundaries are
    the actual frame extrema, so ``train_end < val_end < test_start`` holds.

    Falls back to ``randomSplit`` when the frame carries fewer than 3 distinct
    timestamps (there is no meaningful time ordering to cut on).
    """
    train_ratio = ML_CONFIG["data_split"]["train"]
    val_ratio = ML_CONFIG["data_split"]["validation"]
    test_ratio = ML_CONFIG["data_split"]["test"]
    seed = ML_CONFIG["data_split"]["seed"]
    kind = ML_CONFIG["data_split"].get("kind", "random")

    def _random_fallback():
        train_df, val_df, test_df = df.randomSplit([train_ratio, val_ratio, test_ratio], seed=seed)
        return (
            train_df,
            val_df,
            test_df,
            {
                "kind": "random",
                "train_end": None,
                "val_end": None,
                "test_start": None,
            },
        )

    if kind != "temporal":
        return _random_fallback()

    ts_sorted = [r[0] for r in df.select("timestamp").distinct().orderBy("timestamp").collect()]
    if len(ts_sorted) < 3:
        return _random_fallback()

    n = len(ts_sorted)
    b1 = ts_sorted[int(n * train_ratio)]
    b2 = ts_sorted[int(n * (train_ratio + val_ratio))]

    train_df = df.filter(F.col("timestamp") < b1)
    val_df = df.filter((F.col("timestamp") >= b1) & (F.col("timestamp") < b2))
    test_df = df.filter(F.col("timestamp") >= b2)

    train_end = train_df.agg(F.max("timestamp")).first()[0]
    val_end = val_df.agg(F.max("timestamp")).first()[0]
    test_start = b2

    return (
        train_df,
        val_df,
        test_df,
        {
            "kind": "temporal",
            "train_end": _iso(train_end),
            "val_end": _iso(val_end),
            "test_start": _iso(test_start),
        },
    )


def _build_data_snapshot(df, feature_cols):
    """Rows / time range / city count / feature list of the training frame."""
    stats = df.agg(
        F.min("timestamp").alias("from_ts"),
        F.max("timestamp").alias("to_ts"),
        F.countDistinct("city").alias("cities"),
    ).first()
    return {
        "rows": int(df.count()),
        "from": _iso(stats["from_ts"]),
        "to": _iso(stats["to_ts"]),
        "cities": int(stats["cities"]),
        "features": list(feature_cols),
    }


def _persistence_rmse(df, target_col, observed_col="temperature"):
    """RMSE of the persistence baseline (prediction = current observed value)."""
    rmse = df.select(F.sqrt(F.mean((F.col(target_col) - F.col(observed_col)) ** 2))).first()[0]
    return float(rmse)


def _climatology_rmse(df, test_df, target_col, city_col="city", observed_col="temperature"):
    """RMSE of the per-city train-window mean applied to the test frame.

    A test city absent from the train frame gets no city mean; its residual is
    null and ``F.mean`` drops it, so the climatology RMSE is computed only over
    test cities present in the training window.
    """
    city_means = df.groupBy(city_col).agg(F.mean(observed_col).alias("_city_mean"))
    joined = test_df.join(city_means, on=city_col, how="left")
    rmse = joined.select(F.sqrt(F.mean((F.col(target_col) - F.col("_city_mean")) ** 2))).first()[0]
    return float(rmse)


def _brier_score(df, prob, target_col):
    """Brier score: mean squared error of a probability column vs the binary
    target. ``prob`` is a Column expression (a probability in [0, 1])."""
    return float(df.agg(F.mean((prob - F.col(target_col)) ** 2)).first()[0])


def _persistence_brier(df, rain_col, target_col):
    """Brier score of the persistence baseline: predict "will rain" = "it is
    raining now" (current ``rain`` > 0), matching the target definition in
    ``batch_processing.create_target_variable``."""
    persist = F.when(F.col(rain_col) > 0, 1.0).otherwise(0.0)
    return float(df.agg(F.mean((persist - F.col(target_col)) ** 2)).first()[0])


def _prevalence(df, target_col):
    """Positive rate of the binary target."""
    return float(df.agg(F.mean(F.col(target_col))).first()[0])


def _skill_score(model_score, baseline_score):
    """1 - model/baseline; None when the baseline is 0, missing, or NaN."""
    if baseline_score is None or baseline_score == 0.0:
        return None
    if isinstance(baseline_score, float) and math.isnan(baseline_score):
        return None
    return 1.0 - model_score / baseline_score


def _interval_offsets(residuals, level=0.8):
    """p_lower/p_upper offsets of a ``residual`` column plus their coverage.

    Offsets are the (1-level)/2 and 1-(1-level)/2 quantiles of
    ``(target - prediction)``; coverage is the fraction of residuals inside
    ``[lower_offset, upper_offset]``. Coverage is **in-sample by design**: the
    offsets are the empirical quantiles of the same residuals they are applied
    to, so coverage is ~``level`` by construction. It is reported as a
    calibration diagnostic, not an out-of-sample guarantee.
    """
    lower_q = (1.0 - level) / 2.0
    upper_q = 1.0 - lower_q
    lo, hi = residuals.approxQuantile("residual", [lower_q, upper_q], 0.0)
    total = residuals.count()
    if total == 0:
        coverage = 0.0
    else:
        coverage = (
            residuals.filter((F.col("residual") >= lo) & (F.col("residual") <= hi)).count() / total
        )
    return {
        "level": float(level),
        "lower_offset": float(lo),
        "upper_offset": float(hi),
        "coverage": float(coverage),
    }


# ---------------------------------------------------------------------------
# M4 diagnostics: slices + drift PSI (Contract 3)
# ---------------------------------------------------------------------------

#: Core numeric features the drift PSI covers. These are the physical inputs
#: (the ``save_features_to_mongodb`` dropna subset plus ``specific_humidity``
#: and ``precipitation``); derived lag/rolling/cyclical columns are excluded so
#: the PSI stays interpretable and cheap to compute on the nightly job.
DRIFT_FEATURES = (
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "specific_humidity",
    "precipitation",
)

#: by_hour_of_day emits all 24 local hours; by_rain_bucket all 4 buckets.
HOUR_LABELS = [str(h) for h in range(24)]
RAIN_BUCKET_LABELS = ["dry", "light", "moderate", "heavy"]

#: Rain buckets on observed precipitation (mm): dry <0.1, light 0.1–2,
#: moderate 2–8, heavy >=8.
RAIN_BUCKET_EDGES = [0.1, 2.0, 8.0]


def _to_float(value):
    """Collapse a Spark/NumPy scalar to a finite float, or None."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _grid_config(section: str, horizon: int) -> dict:
    """Param grid for *section*, reduced to one value per parameter (the first)
    for long horizons (``horizon >= ML_CONFIG["long_horizon_from"]``)."""
    config = ML_CONFIG[section]
    if horizon >= ML_CONFIG["long_horizon_from"]:
        return {name: [values[0]] for name, values in config.items()}
    return config


def _psi_bin_index(value, lo, width, n_bins):
    """Bin index for a PSI histogram, clamped to ``[0, n_bins-1]``.

    Values below ``lo`` (or at/above the last edge) are clamped rather than let
    Python's negative-index wraparound send them to the last bin.
    """
    return max(0, min(int((value - lo) / width), n_bins - 1))


def compute_drift_psi(train_df, test_df, features=None):
    """Population Stability Index of *test_df* vs *train_df*, per numeric feature.

    Ten equal-width bins are derived from the **train** distribution (the
    reference); the test distribution is the compared one. Empty bins get an
    ``eps`` guard. A feature is null when it is absent from either frame or has
    too few non-null rows (<10). This is a train→test drift **proxy**, not a
    true PSI (true PSI uses an expected/observed split; here we compare two
    populations with the reference binned first), and is documented as such.
    """
    eps = 1e-6
    min_rows = 10
    n_bins = 10

    features = list(features if features is not None else DRIFT_FEATURES)
    result = {}
    for col in features:
        if col not in train_df.columns or col not in test_df.columns:
            result[col] = None
            continue
        train_vals = [r[0] for r in train_df.select(col).na.drop().collect()]
        test_vals = [r[0] for r in test_df.select(col).na.drop().collect()]
        if len(train_vals) < min_rows or len(test_vals) < min_rows:
            result[col] = None
            continue

        lo = min(train_vals)
        width = (max(train_vals) - lo) / n_bins
        if width <= 0:
            result[col] = None  # constant feature: no meaningful bins
            continue

        train_counts = [0.0] * n_bins
        for v in train_vals:
            train_counts[_psi_bin_index(v, lo, width, n_bins)] += 1.0
        test_counts = [0.0] * n_bins
        for v in test_vals:
            test_counts[_psi_bin_index(v, lo, width, n_bins)] += 1.0

        psi = 0.0
        for i in range(n_bins):
            train_p = max(train_counts[i] / len(train_vals), eps)
            test_p = max(test_counts[i] / len(test_vals), eps)
            psi += (test_p - train_p) * math.log(test_p / train_p)
        result[col] = float(psi)

    return result


def _rain_bucket_column(df):
    """Add ``_rain_bucket`` from observed ``precipitation`` (null-safe)."""
    p = F.col("precipitation")
    return df.withColumn(
        "_rain_bucket",
        F.when(p.isNull(), None)
        .when(p < RAIN_BUCKET_EDGES[0], RAIN_BUCKET_LABELS[0])
        .when(p < RAIN_BUCKET_EDGES[1], RAIN_BUCKET_LABELS[1])
        .when(p < RAIN_BUCKET_EDGES[2], RAIN_BUCKET_LABELS[2])
        .otherwise(RAIN_BUCKET_LABELS[3]),
    )


def _temperature_slice_rows(test_scored, group_col, target_col, labels):
    """Temperature slice rows: rmse/mae/bias per group (brier always null).

    *labels* fixes the emitted cardinality (every label is returned, with
    ``n:0`` and nulls when the group has no rows); ``None`` uses the groups
    actually present in the frame.
    """
    agg = test_scored.groupBy(group_col).agg(
        F.count("*").alias("n"),
        F.sqrt(F.mean((F.col(target_col) - F.col("prediction")) ** 2)).alias("rmse"),
        F.mean(F.abs(F.col(target_col) - F.col("prediction"))).alias("mae"),
        F.mean(F.col("prediction") - F.col(target_col)).alias("bias"),
    )
    index = {str(r[group_col]): r for r in agg.collect()}
    if labels is None:
        labels = sorted(index.keys())
    return [
        {
            "label": str(label),
            "n": int(index[str(label)]["n"]) if str(label) in index else 0,
            "rmse": _to_float(index[str(label)]["rmse"]) if str(label) in index else None,
            "mae": _to_float(index[str(label)]["mae"]) if str(label) in index else None,
            "bias": _to_float(index[str(label)]["bias"]) if str(label) in index else None,
            "brier": None,
        }
        for label in labels
    ]


def _rain_slice_rows(test_scored, group_col, target_col, labels):
    """Rain slice rows: Brier score per group (rmse/mae/bias always null)."""
    pos_prob = vector_to_array(F.col("probability"))[1]
    agg = test_scored.groupBy(group_col).agg(
        F.count("*").alias("n"),
        F.mean((pos_prob - F.col(target_col)) ** 2).alias("brier"),
    )
    index = {str(r[group_col]): r for r in agg.collect()}
    if labels is None:
        labels = sorted(index.keys())
    return [
        {
            "label": str(label),
            "n": int(index[str(label)]["n"]) if str(label) in index else 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "brier": _to_float(index[str(label)]["brier"]) if str(label) in index else None,
        }
        for label in labels
    ]


def build_diagnostics(train_df, test_scored, target_col, kind):
    """M4 diagnostics computed on the temporal test split (``test_scored``).

    ``kind`` selects the metric family: ``"temperature"`` fills rmse/mae/bias
    per slice, ``"rain"`` fills Brier. ``drift_psi`` is the train→test feature
    distribution proxy and is identical for both kinds.
    """
    if kind == "temperature":
        by_city = _temperature_slice_rows(test_scored, "city", target_col, None)
        by_hour = _temperature_slice_rows(test_scored, "hour", target_col, HOUR_LABELS)
        bucketed = _rain_bucket_column(test_scored)
        by_bucket = _temperature_slice_rows(
            bucketed, "_rain_bucket", target_col, RAIN_BUCKET_LABELS
        )
    elif kind == "rain":
        by_city = _rain_slice_rows(test_scored, "city", target_col, None)
        by_hour = _rain_slice_rows(test_scored, "hour", target_col, HOUR_LABELS)
        bucketed = _rain_bucket_column(test_scored)
        by_bucket = _rain_slice_rows(bucketed, "_rain_bucket", target_col, RAIN_BUCKET_LABELS)
    else:
        raise ValueError(f"unknown diagnostics kind: {kind!r}")

    return {
        "by_city": by_city,
        "by_hour_of_day": by_hour,
        "by_rain_bucket": by_bucket,
        "drift_psi": compute_drift_psi(train_df, test_scored),
    }


def train_temperature_prediction_model(df, train_df, val_df, test_df, horizon=1):
    target_col = f"target_temp_{horizon}h"

    assembler, scaler, feature_cols = prepare_features_for_ml(df, target_col, horizon)

    models = {
        "GradientBoostedTrees": GBTRegressor(featuresCol="features", labelCol=target_col),
        "RandomForest": RandomForestRegressor(featuresCol="features", labelCol=target_col),
        "LinearRegression": LinearRegression(featuresCol="features", labelCol=target_col),
    }

    param_grids = {
        "GradientBoostedTrees": ParamGridBuilder()
        .addGrid(
            models["GradientBoostedTrees"].maxDepth,
            _grid_config("gradient_boosted_trees", horizon)["maxDepth"],
        )
        .addGrid(
            models["GradientBoostedTrees"].maxIter,
            _grid_config("gradient_boosted_trees", horizon)["maxIter"],
        )
        .addGrid(
            models["GradientBoostedTrees"].stepSize,
            _grid_config("gradient_boosted_trees", horizon)["stepSize"],
        )
        .build(),
        "RandomForest": ParamGridBuilder()
        .addGrid(
            models["RandomForest"].numTrees,
            _grid_config("random_forest_regressor", horizon)["numTrees"],
        )
        .addGrid(
            models["RandomForest"].maxDepth,
            _grid_config("random_forest_regressor", horizon)["maxDepth"],
        )
        .addGrid(
            models["RandomForest"].minInstancesPerNode,
            _grid_config("random_forest_regressor", horizon)["minInstancesPerNode"],
        )
        .build(),
        "LinearRegression": ParamGridBuilder()
        .addGrid(
            models["LinearRegression"].elasticNetParam,
            _grid_config("linear_regression", horizon)["elasticNetParam"],
        )
        .addGrid(
            models["LinearRegression"].regParam,
            _grid_config("linear_regression", horizon)["regParam"],
        )
        .build(),
    }

    evaluator = RegressionEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="rmse"
    )
    mae_evaluator = RegressionEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="mae"
    )
    r2_evaluator = RegressionEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="r2"
    )

    best_model = None
    best_rmse = float("inf")
    best_model_name = None

    for model_name, model in models.items():
        pipeline = Pipeline(stages=[assembler, scaler, model])

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grids[model_name],
            evaluator=evaluator,
            numFolds=ML_CONFIG["cross_validation"]["num_folds"],
            seed=ML_CONFIG["cross_validation"]["seed"],
            # Fit candidate models concurrently: local mode has one executor, so
            # without this the runner's other cores sit idle.
            parallelism=min(4, os.cpu_count() or 1),
        )

        cv_model = cv.fit(train_df)

        best_pipeline = cv_model.bestModel

        val_predictions = best_pipeline.transform(val_df)
        val_rmse = evaluator.evaluate(val_predictions)

        print(f"\n{model_name} Validation RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = best_pipeline
            best_model_name = model_name

    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"Validation RMSE: {best_rmse:.4f}")

    test_predictions = best_model.transform(test_df)
    # The assembler skips rows with null features (handleInvalid="skip"), so
    # ``test_predictions`` is already a subset of ``test_df``. Score once on a
    # single assembled frame and use it for the model metric AND every baseline
    # so ``skill_score`` never mixes row sets.
    test_scored = test_predictions.filter(F.col("prediction").isNotNull())

    test_rmse = evaluator.evaluate(test_scored)
    test_mae = mae_evaluator.evaluate(test_scored)
    test_r2 = r2_evaluator.evaluate(test_scored)

    print("\n=== FINAL TEST SET PERFORMANCE FOR TEMPERATURE MODEL ===")
    print(f"\nTest RMSE: {test_rmse:.4f}")
    print(f"Test MAE : {test_mae:.4f}")
    print(f"Test R²  : {test_r2:.4f}")

    test_scored.select(
        "city",
        "temperature",
        target_col,
        "prediction",
        F.abs(F.col(target_col) - F.col("prediction")).alias("error"),
    ).show(5)

    if best_model_name in ["GradientBoostedTrees", "RandomForest"]:
        model_stage = best_model.stages[-1]
        feature_importance = model_stage.featureImportances

        important_features = sorted(
            [
                (feature_cols[i], float(importance))
                for i, importance in enumerate(feature_importance)
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:15]

        print("\nTop 15 Important Features:")
        for feature, importance in important_features:
            print(f"  {feature}: {importance:.4f}")

    elif best_model_name == "LinearRegression":
        model_stage = best_model.stages[-1]
        coefficients = model_stage.coefficients.toArray()

        important_features = sorted(
            [(feature_cols[i], abs(float(coef))) for i, coef in enumerate(coefficients)],
            key=lambda x: x[1],
            reverse=True,
        )[:15]

        print("\nTop 15 Important Features (by absolute coefficient):")
        for feature, coef in important_features:
            print(f"  {feature}: {coef:.4f}")

    # The tuned values actually fitted for the winning algorithm (filtered to the
    # grid keys), plus the held-out size for the registry's ``n_test``.
    params = _extract_grid_params(best_model, param_grids[best_model_name])
    n_test = test_scored.count()

    # --- honest metrics on the temporal test split (all on ``test_scored``) ---
    persistence_rmse = _persistence_rmse(test_scored, target_col)
    climatology_rmse = _climatology_rmse(train_df, test_scored, target_col)
    skill_score = _skill_score(test_rmse, persistence_rmse)

    residuals = test_scored.select((F.col(target_col) - F.col("prediction")).alias("residual"))
    interval = _interval_offsets(residuals, level=0.8)
    interval_meta = {
        "level": interval["level"],
        "lower_offset": interval["lower_offset"],
        "upper_offset": interval["upper_offset"],
    }

    diagnostics = build_diagnostics(train_df, test_scored, target_col, "temperature")

    return (
        best_model,
        best_model_name,
        important_features,
        {
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2,
            "persistence_rmse": persistence_rmse,
            "climatology_rmse": climatology_rmse,
            "skill_score": skill_score,
            "coverage": interval["coverage"],
            "n_test": n_test,
        },
        params,
        interval_meta,
        feature_cols,
        diagnostics,
    )


def train_rain_prediction_model(df, train_df, val_df, test_df, horizon=1):
    target_col = f"target_will_rain_{horizon}h"

    assembler, scaler, feature_cols = prepare_features_for_ml(df, target_col, horizon)

    models = {
        "GradientBoostedTrees": GBTClassifier(featuresCol="features", labelCol=target_col),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol=target_col),
    }

    param_grids = {
        "GradientBoostedTrees": ParamGridBuilder()
        .addGrid(
            models["GradientBoostedTrees"].maxDepth,
            _grid_config("gradient_boosted_trees_classifier", horizon)["maxDepth"],
        )
        .addGrid(
            models["GradientBoostedTrees"].maxIter,
            _grid_config("gradient_boosted_trees_classifier", horizon)["maxIter"],
        )
        .addGrid(
            models["GradientBoostedTrees"].stepSize,
            _grid_config("gradient_boosted_trees_classifier", horizon)["stepSize"],
        )
        .build(),
        "RandomForest": ParamGridBuilder()
        .addGrid(
            models["RandomForest"].numTrees,
            _grid_config("random_forest_classifier", horizon)["numTrees"],
        )
        .addGrid(
            models["RandomForest"].maxDepth,
            _grid_config("random_forest_classifier", horizon)["maxDepth"],
        )
        .addGrid(
            models["RandomForest"].minInstancesPerNode,
            _grid_config("random_forest_classifier", horizon)["minInstancesPerNode"],
        )
        .build(),
    }

    evaluator = BinaryClassificationEvaluator(
        labelCol=target_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    best_model = None
    best_auc = 0.0
    best_model_name = None

    for model_name, model in models.items():
        pipeline = Pipeline(stages=[assembler, scaler, model])

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grids[model_name],
            evaluator=evaluator,
            numFolds=ML_CONFIG["cross_validation"]["num_folds"],
            seed=ML_CONFIG["cross_validation"]["seed"],
            # Fit candidate models concurrently: local mode has one executor, so
            # without this the runner's other cores sit idle.
            parallelism=min(4, os.cpu_count() or 1),
        )

        cv_model = cv.fit(train_df)

        best_pipeline = cv_model.bestModel

        val_predictions = best_pipeline.transform(val_df)
        val_auc = evaluator.evaluate(val_predictions)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = best_pipeline
            best_model_name = model_name

    evaluator.setMetricName("areaUnderROC")
    test_predictions = best_model.transform(test_df)
    # Same shared-frame rule as the temperature trainer: score the assembled
    # frame once and run every metric (model + baselines) on that same frame.
    test_scored = test_predictions.filter(F.col("prediction").isNotNull())
    test_auc = evaluator.evaluate(test_scored)
    evaluator.setMetricName("areaUnderPR")
    test_pr_auc = evaluator.evaluate(test_scored)

    print("\n=== FINAL TEST SET PERFORMANCE FOR RAIN MODEL ===")
    print(f"\nTest AUC-ROC: {test_auc:.4f}")
    print(f"\nTest AUC-PR: {test_pr_auc:.4f}")

    print("\nConfusion matrix:")
    test_scored.groupBy(target_col, "prediction").count().show()

    model_stage = best_model.stages[-1]
    feature_importance = model_stage.featureImportances

    important_features = sorted(
        [(feature_cols[i], float(importance)) for i, importance in enumerate(feature_importance)],
        key=lambda x: x[1],
        reverse=True,
    )[:15]

    params = _extract_grid_params(best_model, param_grids[best_model_name])
    n_test = test_scored.count()

    # --- honest metrics on the temporal test split (all on ``test_scored``) ---
    pos_prob = vector_to_array(F.col("probability"))[1]
    brier = _brier_score(test_scored, pos_prob, target_col)
    persistence_brier = _persistence_brier(test_scored, "rain", target_col)
    prevalence = _prevalence(test_scored, target_col)
    skill_score = _skill_score(brier, persistence_brier)

    diagnostics = build_diagnostics(train_df, test_scored, target_col, "rain")

    return (
        best_model,
        best_model_name,
        important_features,
        {
            "auc_roc": test_auc,
            "auc_pr": test_pr_auc,
            "brier": brier,
            "persistence_brier": persistence_brier,
            "prevalence": prevalence,
            "skill_score": skill_score,
            "n_test": n_test,
        },
        params,
        None,
        feature_cols,
        diagnostics,
    )


# ---------------------------------------------------------------------------
# MLflow logging helper
# ---------------------------------------------------------------------------


def _json_safe(value):
    """Coerce a Spark/NumPy scalar to a JSON-serializable Python primitive.

    ``extractParamMap`` and ``featureImportances`` can surface NumPy or Java
    boxed scalars, which PyMongo cannot encode into BSON; collapse them to plain
    int/float/str/list so the registry document stays insertable.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # NumPy arrays (and any object exposing ``tolist`` that is not str/bytes)
    # become plain lists; NumPy scalars fall through to ``item`` below.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _extract_grid_params(best_model, param_maps):
    """Return the fitted values of the parameters that were searched, keyed by
    parameter name and made JSON-serializable."""
    grid_keys = {param.name for param in param_maps[0]}
    fitted = best_model.stages[-1].extractParamMap()
    return {
        param.name: _json_safe(value) for param, value in fitted.items() if param.name in grid_keys
    }


def _log_to_mlflow(
    run_name: str,
    model_type: str,
    model_name: str,
    params: dict,
    metrics: dict,
    important_features: list,
    model,
    horizon: int,
):
    """Log a single training run (temp or rain) to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"weather-{model_type}-prediction")

    with mlflow.start_run(run_name=run_name):
        # --- params ---
        mlflow.log_param("model_algorithm", model_name)
        mlflow.log_param("horizon_hours", horizon)
        mlflow.log_param("train_ratio", ML_CONFIG["data_split"]["train"])
        mlflow.log_param("cv_folds", ML_CONFIG["cross_validation"]["num_folds"])
        for k, v in params.items():
            mlflow.log_param(k, v)

        # --- metrics ---
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

        # --- feature importance as JSON artifact ---
        fi_dict = {f: imp for f, imp in important_features}
        fi_path = f"/tmp/{run_name}_feature_importance.json"
        with open(fi_path, "w") as fh:
            json.dump(fi_dict, fh, indent=2)
        mlflow.log_artifact(fi_path, artifact_path="feature_importance")

        # --- log the Spark PipelineModel ---
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="model",
            registered_model_name=f"weather_{model_type}_{horizon}h",
        )

        print(f"MLflow run '{run_name}' logged successfully.")


def save_model(
    model,
    model_name,
    model_type,
    target,
    horizon,
    metrics=None,
    important_features=None,
    params=None,
    db_name="weather_db",
    metadata_collection="model_registry",
    split=None,
    interval=None,
    data_snapshot=None,
    diagnostics=None,
):
    mongo_url = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
    temp_dir = os.getenv("SPARK_TMP_DIR", "/opt/spark-tmp")
    os.makedirs(temp_dir, exist_ok=True)

    model_dir_path = f"{temp_dir}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_zip_path = f"{model_dir_path}.zip"

    try:
        model.write().overwrite().save(model_dir_path)
        shutil.make_archive(base_name=model_dir_path, format="zip", root_dir=model_dir_path)

        client = MongoClient(mongo_url)
        db = client[db_name]
        fs = GridFS(db)

        with open(model_zip_path, "rb") as model_zip_file:
            file_id = fs.put(model_zip_file, filename=model_zip_path.split("/")[-1])

        print(f"Successfully uploaded model zip to GridFS with file_id: {file_id}")

        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "gridfs_file_id": file_id,
            "timestamp": datetime.now(),
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "target": target,
            "horizon_hours": horizon,
            "stage": "staging",
            "schema_version": 2,
        }

        # Additive fields: the trainers already compute these; persist them so the
        # registry answers "which algorithm won, how well, and why".
        if metrics:
            metadata["metrics"] = {k: _json_safe(v) for k, v in metrics.items()}
        if important_features:
            metadata["feature_importance"] = [
                {"name": name, "importance": _json_safe(importance)}
                for name, importance in important_features
            ]
        if params:
            metadata["params"] = {k: _json_safe(v) for k, v in params.items()}
        if split:
            metadata["split"] = _json_safe(split)
        if interval:
            metadata["interval"] = _json_safe(interval)
        if data_snapshot:
            metadata["data_snapshot"] = _json_safe(data_snapshot)
        if diagnostics:
            metadata["diagnostics"] = _json_safe(diagnostics)
        metadata["commit"] = os.getenv("GITHUB_SHA") or "unknown"

        db[metadata_collection].insert_one(metadata)

    except Exception as e:
        print(f"Error during model and metadata save: {e}")
        raise

    finally:
        if os.path.exists(model_dir_path):
            print(f"Cleaning up temporary directory: {model_dir_path}")
            shutil.rmtree(model_dir_path)
        if os.path.exists(model_zip_path):
            print(f"Cleaning up temporary zip file: {model_zip_path}")
            os.remove(model_zip_path)
        if "client" in locals():
            client.close()


def main():
    spark = create_spark_session("WeatherMLTraining")

    try:
        # Both trainers read the whole feature table (column-presence counts and
        # then the split), so cache it once: without this the run reads Atlas
        # once per horizon, which is a large part of the CI runtime.
        df = load_features(spark).cache()
        df.count()

        horizons = FEATURES_CONFIG["target_horizons"]

        for horizon in horizons:
            print(f"\n=== Training models with {horizon}h prediction horizon ===\n")

            # Drop only this horizon's labels: with several horizons a blanket
            # drop would discard the newest rows for every long horizon.
            label_cols = [f"target_temp_{horizon}h", f"target_will_rain_{horizon}h"]
            horizon_df = df.dropna(subset=label_cols)

            # Split per horizon, then share the frames with both trainers.
            # cache() spills to disk, so it stays safe on the runner's modest
            # driver heap, and avoids re-reading / re-sorting per candidate fit.
            train_df, val_df, test_df, split_meta = temporal_split(horizon_df)
            train_df = train_df.cache()
            val_df = val_df.cache()
            test_df = test_df.cache()
            print(
                f"Train size: {train_df.count()}, "
                f"Validation size: {val_df.count()}, "
                f"Test size: {test_df.count()}"
            )

            (
                temp_model,
                temp_model_name,
                temp_features,
                temp_metrics,
                temp_params,
                temp_interval,
                temp_feature_cols,
                temp_diagnostics,
            ) = train_temperature_prediction_model(horizon_df, train_df, val_df, test_df, horizon)
            (
                rain_model,
                rain_model_name,
                rain_features,
                rain_metrics,
                rain_params,
                _,
                _,
                rain_diagnostics,
            ) = train_rain_prediction_model(horizon_df, train_df, val_df, test_df, horizon)

            # One snapshot of this horizon's training frame; both models were
            # trained on the same rows and the same feature list.
            data_snapshot = _build_data_snapshot(horizon_df, temp_feature_cols)

            # --- persist to GridFS (used by inference.py) ---
            save_model(
                temp_model,
                f"temp_prediction_{horizon}h_{temp_model_name}",
                model_type=temp_model_name,
                target="temperature",
                horizon=horizon,
                metrics=temp_metrics,
                important_features=temp_features,
                params=temp_params,
                split=split_meta,
                interval=temp_interval,
                data_snapshot=data_snapshot,
                diagnostics=temp_diagnostics,
            )
            save_model(
                rain_model,
                f"rain_prediction_{horizon}h_{rain_model_name}",
                model_type=rain_model_name,
                target="rain",
                horizon=horizon,
                metrics=rain_metrics,
                important_features=rain_features,
                params=rain_params,
                split=split_meta,
                data_snapshot=data_snapshot,
                diagnostics=rain_diagnostics,
            )

            # --- log to MLflow ---
            _log_to_mlflow(
                run_name=f"temp_{temp_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type="temperature",
                model_name=temp_model_name,
                params=temp_params,
                metrics=temp_metrics,
                important_features=temp_features,
                model=temp_model,
                horizon=horizon,
            )
            _log_to_mlflow(
                run_name=f"rain_{rain_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type="rain",
                model_name=rain_model_name,
                params=rain_params,
                metrics=rain_metrics,
                important_features=rain_features,
                model=rain_model,
                horizon=horizon,
            )

            # Free the per-horizon split frames before the next horizon.
            for frame in (train_df, val_df, test_df):
                frame.unpersist()

        # Scoring is inference.py's job: it writes the ``source_timestamp`` /
        # ``predicted_temperature`` schema the API reads. ml_training used to
        # append a second, incompatible document shape here, which the backend
        # silently ignored.

    except Exception as e:
        print(f"Error during ML training: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
