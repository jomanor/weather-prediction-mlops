"""
tests/unit/test_ml_training.py
==============================
Unit tests for spark/spark-jobs/ml_training.py.

Covers the temporal split helper, the honest-metric baselines (persistence /
climatology RMSE, Brier, prevalence), the interval offsets, ``_json_safe`` and
the ``save_model`` registry document, all on a tiny synthetic Spark DataFrame
with a local master. No MongoDB, no MLflow, no cluster.
"""

import importlib.util
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

SPARK_JOBS_DIR = os.path.join(os.path.dirname(__file__), "../../spark/spark-jobs")
SPARK_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../spark/config")
for d in (SPARK_JOBS_DIR, SPARK_CONFIG_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import ml_training  # noqa: E402


def _load_real_ml_config():
    """Load the real ``ML_CONFIG`` from spark/config/spark_config.py on disk.

    ``test_batch_processing`` stubs ``spark_config`` in ``sys.modules``, so
    importing ``ml_training`` may bind its ``ML_CONFIG`` to a MagicMock. Load
    the real dict from the file directly (under a private module name) so the
    split tests are order-independent.
    """
    spec = importlib.util.spec_from_file_location(
        "_ml_training_real_spark_config", os.path.join(SPARK_CONFIG_DIR, "spark_config.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ML_CONFIG


@pytest.fixture()
def ml_config(monkeypatch):
    """Patch ``ml_training.ML_CONFIG`` to the real config for this test."""
    real = _load_real_ml_config()
    monkeypatch.setattr(ml_training, "ML_CONFIG", real)
    return real


def _split_schema():
    return StructType(
        [
            StructField("city", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("target_temp_1h", DoubleType(), True),
        ]
    )


def _multi_city_df(spark):
    """3 cities, 5 hourly timestamps each (every city shares each hour)."""
    cities = ["Madrid", "Valencia", "Sevilla"]
    rows = [
        Row(
            city=city,
            timestamp=datetime(2026, 1, 1, h, 0, 0),
            temperature=float(h),
            target_temp_1h=float(h + 1),
        )
        for city in cities
        for h in range(5)
    ]
    return spark.createDataFrame(rows, schema=_split_schema())


class TestTemporalSplit:
    def test_temporal_split_exact_boundaries_no_shared_timestamp(self, spark, ml_config):
        df = _multi_city_df(spark)
        train_df, val_df, test_df, split = ml_training.temporal_split(df)

        assert split["kind"] == "temporal"
        # 5 distinct timestamps, ratios 0.6/0.2/0.2 -> b1 = ts[3], b2 = ts[4].
        assert split["train_end"] == "2026-01-01T02:00:00"
        assert split["val_end"] == "2026-01-01T03:00:00"
        assert split["test_start"] == "2026-01-01T04:00:00"
        assert split["train_end"] < split["val_end"] < split["test_start"]

        # 3 cities x 3h train, 3 cities x 1h val, 3 cities x 1h test.
        assert train_df.count() == 9
        assert val_df.count() == 3
        assert test_df.count() == 3

        train_ts = {r[0] for r in train_df.select("timestamp").distinct().collect()}
        val_ts = {r[0] for r in val_df.select("timestamp").distinct().collect()}
        test_ts = {r[0] for r in test_df.select("timestamp").distinct().collect()}
        assert train_ts.isdisjoint(val_ts)
        assert val_ts.isdisjoint(test_ts)
        assert train_ts.isdisjoint(test_ts)

    def test_temporal_split_deterministic(self, spark, ml_config):
        df = _multi_city_df(spark)
        split_a = ml_training.temporal_split(df)
        split_b = ml_training.temporal_split(df)

        assert split_a[3] == split_b[3]
        assert split_a[0].count() == split_b[0].count()
        assert split_a[1].count() == split_b[1].count()
        assert split_a[2].count() == split_b[2].count()

    def test_temporal_split_falls_back_when_few_timestamps(self, spark, ml_config):
        # Only two distinct timestamps -> no meaningful time ordering -> random.
        rows = [
            Row(
                city="Madrid",
                timestamp=datetime(2026, 1, 1, 1, 0, 0),
                temperature=1.0,
                target_temp_1h=2.0,
            ),
            Row(
                city="Madrid",
                timestamp=datetime(2026, 1, 1, 2, 0, 0),
                temperature=2.0,
                target_temp_1h=3.0,
            ),
            Row(
                city="Valencia",
                timestamp=datetime(2026, 1, 1, 1, 0, 0),
                temperature=5.0,
                target_temp_1h=6.0,
            ),
            Row(
                city="Valencia",
                timestamp=datetime(2026, 1, 1, 2, 0, 0),
                temperature=6.0,
                target_temp_1h=7.0,
            ),
        ]
        df = spark.createDataFrame(rows, schema=_split_schema())

        train_df, val_df, test_df, split = ml_training.temporal_split(df)

        assert split["kind"] == "random"
        assert split["train_end"] is None
        assert split["val_end"] is None
        assert split["test_start"] is None
        # randomSplit partitions rows: counts sum to the total, no shared row.
        assert train_df.count() + val_df.count() + test_df.count() == df.count()


class TestTemperatureBaselines:
    def test_persistence_rmse(self, spark):
        # residuals (target - temperature) = [1, 2, 2] -> rmse = sqrt(9/3)
        rows = [
            Row(temperature=1.0, target_temp_1h=2.0),
            Row(temperature=2.0, target_temp_1h=4.0),
            Row(temperature=3.0, target_temp_1h=5.0),
        ]
        schema = StructType(
            [
                StructField("temperature", DoubleType(), True),
                StructField("target_temp_1h", DoubleType(), True),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)

        rmse = ml_training._persistence_rmse(df, "target_temp_1h")
        assert rmse == pytest.approx(3.0**0.5)

    def test_climatology_rmse(self, spark):
        train = spark.createDataFrame(
            [
                Row(city="A", temperature=10.0),
                Row(city="A", temperature=20.0),  # A mean = 15
                Row(city="B", temperature=0.0),  # B mean = 0
            ],
            schema=StructType(
                [
                    StructField("city", StringType(), True),
                    StructField("temperature", DoubleType(), True),
                ]
            ),
        )
        test = spark.createDataFrame(
            [
                Row(city="A", target_temp_1h=16.0),  # residual 1
                Row(city="B", target_temp_1h=2.0),  # residual 2
            ],
            schema=StructType(
                [
                    StructField("city", StringType(), True),
                    StructField("target_temp_1h", DoubleType(), True),
                ]
            ),
        )

        rmse = ml_training._climatology_rmse(train, test, "target_temp_1h")
        # residuals [1, 2] -> rmse = sqrt((1 + 4) / 2)
        assert rmse == pytest.approx(2.5**0.5)

    def test_climatology_rmse_drops_city_absent_from_train(self, spark):
        # City "B" exists in test but not train -> its residual is null and
        # F.mean ignores it, so only city "A"'s rows contribute.
        train = spark.createDataFrame(
            [Row(city="A", temperature=10.0), Row(city="A", temperature=20.0)],
            schema=StructType(
                [
                    StructField("city", StringType(), True),
                    StructField("temperature", DoubleType(), True),
                ]
            ),
        )
        test = spark.createDataFrame(
            [Row(city="A", target_temp_1h=16.0), Row(city="B", target_temp_1h=999.0)],
            schema=StructType(
                [
                    StructField("city", StringType(), True),
                    StructField("target_temp_1h", DoubleType(), True),
                ]
            ),
        )

        rmse = ml_training._climatology_rmse(train, test, "target_temp_1h")
        # Only city A: residual = 16 - 15 = 1 -> rmse = 1.0
        assert rmse == pytest.approx(1.0)


class TestRainBaselines:
    def test_brier_score(self, spark):
        rows = [Row(prob=0.1, target=0.0), Row(prob=0.9, target=1.0)]
        schema = StructType(
            [
                StructField("prob", DoubleType(), True),
                StructField("target", DoubleType(), True),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)
        # (0.1-0)^2 + (0.9-1)^2 = 0.01 + 0.01 = 0.02 / 2 = 0.01
        assert ml_training._brier_score(df, F.col("prob"), "target") == pytest.approx(0.01)

    def test_persistence_brier(self, spark):
        rows = [
            Row(rain=0.0, target=0.0),
            Row(rain=1.0, target=0.0),
            Row(rain=1.0, target=1.0),
        ]
        schema = StructType(
            [
                StructField("rain", DoubleType(), True),
                StructField("target", DoubleType(), True),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)
        # persistence predictions [0, 1, 1] vs target [0, 0, 1] -> errors [0, 1, 0]
        assert ml_training._persistence_brier(df, "rain", "target") == pytest.approx(1.0 / 3.0)

    def test_prevalence(self, spark):
        rows = [Row(target=0.0), Row(target=1.0), Row(target=1.0)]
        schema = StructType([StructField("target", DoubleType(), True)])
        df = spark.createDataFrame(rows, schema=schema)
        assert ml_training._prevalence(df, "target") == pytest.approx(2.0 / 3.0)


class TestSkillScore:
    def test_skill_score(self):
        assert ml_training._skill_score(1.0, 2.0) == pytest.approx(0.5)

    def test_skill_score_undefined_when_baseline_zero(self):
        assert ml_training._skill_score(1.0, 0.0) is None

    def test_skill_score_undefined_when_baseline_nan(self):
        assert ml_training._skill_score(1.0, float("nan")) is None


class TestIntervalOffsets:
    def test_interval_offsets_p10_p90_and_coverage(self, spark):
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        schema = StructType([StructField("residual", DoubleType(), True)])
        df = spark.createDataFrame([Row(residual=v) for v in vals], schema=schema)

        result = ml_training._interval_offsets(df, level=0.8)

        assert result["level"] == 0.8
        # Spark 3.5 exact p10/p90 of [0..9] are index-based: 0.0 and 8.0.
        assert result["lower_offset"] == pytest.approx(0.0)
        assert result["upper_offset"] == pytest.approx(8.0)
        # fraction of residuals within [0.0, 8.0] = 9 / 10
        assert result["coverage"] == pytest.approx(0.9)


class TestReducedGrid:
    def test_long_horizon_uses_first_value_of_each_param(self, ml_config):
        # horizon < long_horizon_from (12) -> full grid.
        assert ml_training._grid_config("gradient_boosted_trees", 1)["maxDepth"] == [5, 7]
        assert ml_training._grid_config("random_forest_regressor", 6)["numTrees"] == [40, 80]
        assert ml_training._grid_config("linear_regression", 6)["regParam"] == [0.01, 0.1]

        # horizon >= long_horizon_from -> one value per parameter.
        assert ml_training._grid_config("gradient_boosted_trees", 12)["maxDepth"] == [5]
        assert ml_training._grid_config("gradient_boosted_trees", 24)["maxIter"] == [50]
        assert ml_training._grid_config("random_forest_regressor", 12)["numTrees"] == [40]
        assert ml_training._grid_config("gradient_boosted_trees_classifier", 24)["stepSize"] == [
            0.1
        ]


class TestDriftPsi:
    def test_identical_distributions_yield_zero_psi(self, spark):
        vals = [float(i) for i in range(20)]
        schema = StructType([StructField("temperature", DoubleType(), True)])
        df = spark.createDataFrame([Row(temperature=v) for v in vals], schema=schema)

        psi = ml_training.compute_drift_psi(df, df, features=["temperature"])
        assert psi["temperature"] == pytest.approx(0.0)

    def test_missing_and_few_rows_are_null(self, spark):
        schema = StructType([StructField("temperature", DoubleType(), True)])
        train = spark.createDataFrame([Row(temperature=float(i)) for i in range(20)], schema=schema)
        test_few = spark.createDataFrame([Row(temperature=1.0)], schema=schema)

        psi = ml_training.compute_drift_psi(train, test_few, features=["temperature", "humidity"])
        # too few test rows -> null; absent column -> null.
        assert psi["temperature"] is None
        assert psi["humidity"] is None

    def test_shifted_distribution_yields_positive_psi(self, spark):
        schema = StructType([StructField("temperature", DoubleType(), True)])
        train = spark.createDataFrame([Row(temperature=float(i)) for i in range(20)], schema=schema)
        test = spark.createDataFrame(
            [Row(temperature=float(i + 50)) for i in range(20)], schema=schema
        )

        psi = ml_training.compute_drift_psi(train, test, features=["temperature"])
        assert psi["temperature"] > 0.0

    def test_bin_index_clamps_below_train_min_to_zero(self):
        # v < lo must land in bin 0, not wrap to the last bin via negative index.
        assert ml_training._psi_bin_index(5.0, 10.0, 2.0, 10) == 0
        assert ml_training._psi_bin_index(10.0, 10.0, 2.0, 10) == 0  # exactly lo
        assert ml_training._psi_bin_index(11.9, 10.0, 2.0, 10) == 0
        assert ml_training._psi_bin_index(20.0, 10.0, 2.0, 10) == 5
        assert ml_training._psi_bin_index(100.0, 10.0, 2.0, 10) == 9  # above last edge


def _multi_horizon_frame(spark, null_newest_targets=False):
    """A frame carrying all five horizons' target columns plus observation
    inputs. With ``null_newest_targets`` the newest row (max timestamp) has
    every ``target_*`` column null — exactly the latest-features shape at
    inference time."""
    horizons = (1, 3, 6, 12, 24)
    fields = [
        StructField("city", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("temperature", DoubleType(), True),
        StructField("humidity", DoubleType(), True),
        StructField("pressure", DoubleType(), True),
        StructField("wind_speed", DoubleType(), True),
    ]
    for h in horizons:
        fields.append(StructField(f"target_temp_{h}h", DoubleType(), True))
        fields.append(StructField(f"target_will_rain_{h}h", DoubleType(), True))
    schema = StructType(fields)

    n = 15
    rows = []
    for i in range(n):
        newest = i == n - 1
        row = {
            "city": "Madrid",
            "timestamp": datetime(2026, 1, 1, i, 0, 0),
            "temperature": float(i),
            "humidity": 50.0,
            "pressure": 1013.0,
            "wind_speed": 5.0,
        }
        for h in horizons:
            temp = None if (null_newest_targets and newest) else float(i + h)
            rain = None if temp is None else (1.0 if (i + h) % 2 else 0.0)
            row[f"target_temp_{h}h"] = temp
            row[f"target_will_rain_{h}h"] = rain
        rows.append(Row(**row))
    return spark.createDataFrame(rows, schema=schema)


class TestFeatureExclusionLeakage:
    def test_no_target_column_becomes_a_feature(self, spark):
        """Every horizon's ``target_*`` column must be excluded, not just the
        current one — a future-horizon label would leak the true future."""
        df = _multi_horizon_frame(spark, null_newest_targets=True)
        for h in (1, 3, 6, 12, 24):
            _, _, feature_cols = ml_training.prepare_features_for_ml(df, f"target_temp_{h}h", h)
            assert feature_cols, "expected at least one usable feature column"
            assert not any(
                c.startswith("target_") for c in feature_cols
            ), f"horizon {h}h leaked a target column into features: {feature_cols}"

    def test_newest_row_with_null_targets_still_assembles(self, spark):
        """The newest feature row has every ``target_*`` null; once targets are
        excluded the assembler must not drop it (inference would upsert zero
        rows otherwise)."""
        df = _multi_horizon_frame(spark, null_newest_targets=True)
        assembler, scaler, feature_cols = ml_training.prepare_features_for_ml(
            df, "target_temp_1h", 1
        )
        assert not any(c.startswith("target_") for c in feature_cols)

        assembled = assembler.transform(df)
        scaler_model = scaler.fit(assembled)
        scaled = scaler_model.transform(assembled)

        newest = scaled.orderBy(F.col("timestamp").desc()).first()
        assert newest["features_raw"] is not None
        assert newest["features_raw"].size == len(feature_cols)
        assert newest["features"] is not None


def _diag_temp_frame(spark):
    schema = StructType(
        [
            StructField("city", StringType(), True),
            StructField("hour", IntegerType(), True),
            StructField("precipitation", DoubleType(), True),
            StructField("target_temp_1h", DoubleType(), True),
            StructField("prediction", DoubleType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("humidity", DoubleType(), True),
            StructField("pressure", DoubleType(), True),
            StructField("wind_speed", DoubleType(), True),
            StructField("specific_humidity", DoubleType(), True),
        ]
    )
    rows = []
    for i in range(20):
        # i % 4 == 0 -> precipitation 5.0 (moderate), else dry. Hours 0..19 only,
        # so hours 20..23 are empty slices.
        rows.append(
            Row(
                city="Madrid" if i < 10 else "Valencia",
                hour=i % 24,
                precipitation=5.0 if i % 4 == 0 else 0.0,
                target_temp_1h=20.0 + i * 0.5,
                prediction=21.0 + i * 0.5,  # constant +1.0 over-forecast
                temperature=float(i),
                humidity=float(50 + i % 10),
                pressure=float(1013 + i % 5),
                wind_speed=5.0,
                specific_humidity=0.01,
            )
        )
    return spark.createDataFrame(rows, schema=schema)


class TestBuildDiagnostics:
    def test_temperature_diagnostics_shape(self, spark):
        frame = _diag_temp_frame(spark)
        diag = ml_training.build_diagnostics(frame, frame, "target_temp_1h", "temperature")

        assert set(diag) == {"by_city", "by_hour_of_day", "by_rain_bucket", "drift_psi"}
        assert len(diag["by_hour_of_day"]) == 24
        assert [s["label"] for s in diag["by_hour_of_day"]] == [str(h) for h in range(24)]
        assert [s["label"] for s in diag["by_rain_bucket"]] == ["dry", "light", "moderate", "heavy"]

        # Every slice carries the SliceMetric keys.
        for slices in (diag["by_city"], diag["by_hour_of_day"], diag["by_rain_bucket"]):
            for s in slices:
                assert set(s) == {"label", "n", "rmse", "mae", "bias", "brier"}

        # Temperature slices fill rmse/mae/bias (constant +1.0 error), brier null.
        madrid = next(s for s in diag["by_city"] if s["label"] == "Madrid")
        assert madrid["n"] == 10
        assert madrid["rmse"] == pytest.approx(1.0)
        assert madrid["mae"] == pytest.approx(1.0)
        assert madrid["bias"] == pytest.approx(1.0)
        assert madrid["brier"] is None

        # Empty slices are emitted with n:0 and nulls.
        empty_hour = next(s for s in diag["by_hour_of_day"] if s["label"] == "23")
        assert empty_hour == {
            "label": "23",
            "n": 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "brier": None,
        }
        heavy = next(s for s in diag["by_rain_bucket"] if s["label"] == "heavy")
        assert heavy == {
            "label": "heavy",
            "n": 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "brier": None,
        }

        # drift_psi covers the core numeric features.
        assert set(diag["drift_psi"]) == set(ml_training.DRIFT_FEATURES)

    def test_rain_diagnostics_fill_brier_only(self, spark):
        schema = StructType(
            [
                StructField("city", StringType(), True),
                StructField("hour", IntegerType(), True),
                StructField("precipitation", DoubleType(), True),
                StructField("target_will_rain_1h", DoubleType(), True),
                StructField("probability", VectorUDT(), True),
                StructField("temperature", DoubleType(), True),
                StructField("humidity", DoubleType(), True),
                StructField("pressure", DoubleType(), True),
                StructField("wind_speed", DoubleType(), True),
                StructField("specific_humidity", DoubleType(), True),
            ]
        )
        rows = [
            Row(
                city="Madrid",
                hour=6,
                precipitation=0.0,
                target_will_rain_1h=1.0,
                probability=Vectors.dense([0.1, 0.9]),  # p=0.9 -> brier (0.9-1)^2 = 0.01
                temperature=float(i),
                humidity=50.0,
                pressure=1013.0,
                wind_speed=5.0,
                specific_humidity=0.01,
            )
            for i in range(12)
        ]
        frame = spark.createDataFrame(rows, schema=schema)

        diag = ml_training.build_diagnostics(frame, frame, "target_will_rain_1h", "rain")

        madrid = next(s for s in diag["by_city"] if s["label"] == "Madrid")
        assert madrid["n"] == 12
        assert madrid["brier"] == pytest.approx(0.01)
        assert madrid["rmse"] is None
        assert madrid["mae"] is None
        assert madrid["bias"] is None

    def test_unknown_kind_rejected(self, spark):
        frame = _diag_temp_frame(spark)
        with pytest.raises(ValueError):
            ml_training.build_diagnostics(frame, frame, "target_temp_1h", "bogus")


class TestJsonSafe:
    def test_numpy_float_int_bool(self):
        assert ml_training._json_safe(np.float64(1.5)) == 1.5
        assert isinstance(ml_training._json_safe(np.float64(1.5)), float)
        assert ml_training._json_safe(np.int64(3)) == 3
        assert isinstance(ml_training._json_safe(np.int64(3)), int)
        assert ml_training._json_safe(np.bool_(True)) is True

    def test_numpy_ndarray_to_list(self):
        result = ml_training._json_safe(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_nested_dict_list(self):
        value = {"a": [1, {"b": (2, 3)}], "c": 4}
        assert ml_training._json_safe(value) == {"a": [1, {"b": [2, 3]}], "c": 4}

    def test_item_dunder_scalar(self):
        # A Java/NumPy boxed scalar surfaced with an ``item`` method.
        class _Boxed:
            def item(self):
                return 7

        assert ml_training._json_safe(_Boxed()) == 7

    def test_plain_passthrough(self):
        assert ml_training._json_safe("x") == "x"
        assert ml_training._json_safe(None) is None
        assert ml_training._json_safe(42) == 42


class _FakeWritableModel:
    def write(self):
        return self

    def overwrite(self):
        return self

    def save(self, path):
        return None


def _save_and_capture(monkeypatch, tmp_path, commit, **save_kwargs):
    """Run ``save_model`` against mocked Mongo/GridFS and return the doc inserted."""
    monkeypatch.setenv("SPARK_TMP_DIR", str(tmp_path))
    monkeypatch.setenv("MONGO_URI", "mongodb://fake")
    monkeypatch.setenv("GITHUB_SHA", commit)

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.__getitem__.return_value = mock_db

    with (
        patch("ml_training.MongoClient", return_value=mock_client),
        patch("ml_training.GridFS"),
        patch("ml_training.shutil.make_archive"),
        patch("builtins.open", mock_open(read_data=b"z")),
    ):
        ml_training.save_model(
            _FakeWritableModel(), "test_model", "GBT", "temperature", 1, **save_kwargs
        )

    return mock_db["model_registry"].insert_one.call_args[0][0]


class TestSaveModel:
    def test_save_model_persists_additive_metadata_bson_safe(self, monkeypatch, tmp_path):
        split = {
            "kind": "temporal",
            "train_end": "2026-01-01T00:00:00",
            "val_end": "2026-01-02T00:00:00",
            "test_start": "2026-01-03T00:00:00",
        }
        interval = {"level": 0.8, "lower_offset": -2.0, "upper_offset": 2.5}
        snapshot = {
            "rows": 100,
            "from": "2026-01-01T00:00:00",
            "to": "2026-01-05T00:00:00",
            "cities": 3,
            "features": ["temperature"],
        }

        inserted = _save_and_capture(
            monkeypatch,
            tmp_path,
            "abc123",
            metrics={"rmse": 1.5, "skill_score": np.float64(0.5), "coverage": 0.8, "n_test": 10},
            important_features=[("temperature", 0.9)],
            params={"maxDepth": 5},
            split=split,
            interval=interval,
            data_snapshot=snapshot,
        )

        assert inserted["commit"] == "abc123"
        assert inserted["metrics"]["skill_score"] == 0.5
        assert isinstance(inserted["metrics"]["skill_score"], float)
        assert inserted["metrics"]["n_test"] == 10
        assert inserted["split"] == split
        assert inserted["interval"] == interval
        assert inserted["data_snapshot"] == snapshot
        assert inserted["feature_importance"] == [{"name": "temperature", "importance": 0.9}]
        assert inserted["params"] == {"maxDepth": 5}

    def test_save_model_commit_falls_back_to_unknown(self, monkeypatch, tmp_path):
        inserted = _save_and_capture(monkeypatch, tmp_path, "")
        assert inserted["commit"] == "unknown"

    def test_save_model_persists_diagnostics(self, monkeypatch, tmp_path):
        diagnostics = {
            "by_city": [
                {"label": "Madrid", "n": 10, "rmse": 1.0, "mae": 1.0, "bias": 1.0, "brier": None}
            ],
            "by_hour_of_day": [
                {"label": "0", "n": 0, "rmse": None, "mae": None, "bias": None, "brier": None}
            ],
            "by_rain_bucket": [
                {"label": "dry", "n": 4, "rmse": 1.0, "mae": 1.0, "bias": 1.0, "brier": None}
            ],
            "drift_psi": {"temperature": 0.08, "humidity": None},
        }
        inserted = _save_and_capture(monkeypatch, tmp_path, "abc123", diagnostics=diagnostics)
        assert inserted["diagnostics"] == diagnostics


class _FakeReader:
    """Chains the mongodb reader options and returns a fixed DataFrame."""

    def __init__(self, df):
        self._df = df

    def format(self, *_args, **_kwargs):
        return self

    def option(self, *_args, **_kwargs):
        return self

    def load(self):
        return self._df


class _FakeSpark:
    def __init__(self, df):
        self.read = _FakeReader(df)


class TestLoadFeatures:
    """``load_features`` drops on the observation inputs only, so the newest
    feature row (all horizon targets null) survives — the freshness regression."""

    def test_drops_incomplete_observations_and_dedupes_but_keeps_null_targets(self, spark):
        schema = StructType(
            [
                StructField("_id", StringType(), True),
                StructField("city", StringType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("temperature", DoubleType(), True),
                StructField("target_temp_1h", DoubleType(), True),
                StructField("target_temp_3h", DoubleType(), True),
            ]
        )
        base = datetime(2026, 1, 1, 1, 0, 0)
        rows = [
            # Duplicate (city, timestamp): an overlapping run wrote it twice.
            Row(
                _id="a",
                city="Madrid",
                timestamp=base,
                temperature=10.0,
                target_temp_1h=11.0,
                target_temp_3h=13.0,
            ),
            Row(
                _id="b",
                city="Madrid",
                timestamp=base,
                temperature=10.0,
                target_temp_1h=11.0,
                target_temp_3h=13.0,
            ),
            # No observation -> dropped regardless of targets.
            Row(
                _id="c",
                city="Madrid",
                timestamp=base + timedelta(hours=1),
                temperature=None,
                target_temp_1h=12.0,
                target_temp_3h=14.0,
            ),
            # Newest row: observation present, every target null -> must be kept.
            Row(
                _id="d",
                city="Valencia",
                timestamp=base + timedelta(hours=2),
                temperature=30.0,
                target_temp_1h=None,
                target_temp_3h=None,
            ),
        ]
        df = spark.createDataFrame(rows, schema=schema)

        out = ml_training.load_features(_FakeSpark(df)).orderBy("timestamp").collect()

        assert len(out) == 2
        # The connector's `_id` is dropped before training/registry use.
        assert "_id" not in out[0].asDict()
        # Deduped to one row per (city, timestamp) ...
        assert out[0].city == "Madrid" and out[0].temperature == 10.0
        # ... and the newest all-null-target row is retained.
        assert out[1].timestamp == base + timedelta(hours=2)
        assert out[1].target_temp_1h is None and out[1].target_temp_3h is None


def _horizon_isolation_frame(spark):
    """Each horizon has a different block of missing labels: 1h labels are null
    on the first 3 rows, 3h labels on the last 3. A per-horizon dropna must
    discard exactly those rows, so the two horizons train on different windows."""
    schema = StructType(
        [
            StructField("city", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("target_temp_1h", DoubleType(), True),
            StructField("target_will_rain_1h", DoubleType(), True),
            StructField("target_temp_3h", DoubleType(), True),
            StructField("target_will_rain_3h", DoubleType(), True),
        ]
    )
    base = datetime(2026, 1, 1, 0, 0, 0)
    rows = []
    for i in range(10):
        one_missing = i < 3
        three_missing = i >= 7
        rows.append(
            Row(
                city="Madrid",
                timestamp=base + timedelta(hours=i),
                temperature=float(i),
                target_temp_1h=None if one_missing else float(i + 1),
                target_will_rain_1h=None if one_missing else float(i % 2),
                target_temp_3h=None if three_missing else float(i + 3),
                target_will_rain_3h=None if three_missing else float(i % 2),
            )
        )
    return spark.createDataFrame(rows, schema=schema)


class TestMainPerHorizonIsolation:
    """``main()`` drops only the current horizon's labels before training and
    saves that horizon's models — a blanket dropna would wipe the newest rows."""

    def test_each_horizon_drops_only_its_own_labels(self, spark, monkeypatch):
        frame = _horizon_isolation_frame(spark)
        trained: dict[int, list] = {}
        saved: list[tuple] = []

        monkeypatch.setattr(ml_training, "FEATURES_CONFIG", {"target_horizons": [1, 3]})
        monkeypatch.setattr(ml_training, "create_spark_session", lambda *a, **k: MagicMock())
        monkeypatch.setattr(ml_training, "load_features", lambda _spark: frame)

        def fake_split(horizon_df):
            return horizon_df, horizon_df, horizon_df, {"kind": "temporal"}

        monkeypatch.setattr(ml_training, "temporal_split", fake_split)

        def fake_temp_train(horizon_df, _train, _val, _test, horizon):
            trained.setdefault(horizon, []).append(horizon_df)
            return (
                MagicMock(),
                "GBT",
                [],
                {"rmse": 1.0},
                {},
                {},
                ["temperature"],
                {"drift_psi": {}},
            )

        def fake_rain_train(_df, _train, _val, _test, horizon):
            return (MagicMock(), "GBT", [], {"brier": 0.1}, {}, None, ["temperature"], None)

        monkeypatch.setattr(ml_training, "train_temperature_prediction_model", fake_temp_train)
        monkeypatch.setattr(ml_training, "train_rain_prediction_model", fake_rain_train)

        def fake_save(_model, name, **kwargs):
            saved.append((name, kwargs["horizon"]))

        monkeypatch.setattr(ml_training, "save_model", fake_save)
        monkeypatch.setattr(ml_training, "_log_to_mlflow", lambda **kwargs: None)

        ml_training.main()

        # Per-horizon frame: 1h keeps rows 3..9, 3h keeps rows 0..6.
        h1_df, h3_df = trained[1][0], trained[3][0]
        assert h1_df.count() == 7
        assert h3_df.count() == 7
        assert h1_df.agg(F.min("timestamp")).first()[0] == datetime(2026, 1, 1, 3, 0, 0)
        assert h3_df.agg(F.min("timestamp")).first()[0] == datetime(2026, 1, 1, 0, 0, 0)
        # No label of the horizon being trained is null (dropna took effect).
        assert h1_df.filter(F.col("target_temp_1h").isNull()).count() == 0
        assert h1_df.filter(F.col("target_will_rain_1h").isNull()).count() == 0
        assert h3_df.filter(F.col("target_temp_3h").isNull()).count() == 0
        # Both horizons saved a temp and a rain model, tagged with their horizon.
        assert sorted(saved) == [
            ("rain_prediction_1h_GBT", 1),
            ("rain_prediction_3h_GBT", 3),
            ("temp_prediction_1h_GBT", 1),
            ("temp_prediction_3h_GBT", 3),
        ]
