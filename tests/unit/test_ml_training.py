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
from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
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
