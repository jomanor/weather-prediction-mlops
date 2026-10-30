"""
test_inference.py
=================
Unit tests for spark-jobs/inference.py.
Tests model loading, registry queries, and batch inference transformations.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from functools import partial
from unittest.mock import MagicMock, call, patch

import pytest
from pyspark.sql import DataFrameWriter, Row
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from tests.unit.fake_registry import FakeDb, FakeRegistryCollection, dt, make_entry

# Ensure spark jobs and config are on sys.path
SPARK_JOBS_DIR = os.path.join(os.path.dirname(__file__), "../../spark/spark-jobs")
SPARK_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../spark/config")
for d in (SPARK_JOBS_DIR, SPARK_CONFIG_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import inference  # noqa: E402


class TestModelRegistryLookup:
    """Registry resolution is stage-aware (M5) and lives in ``model_selection``."""

    def test_production_is_preferred_over_a_newer_staging_doc(self):
        db = FakeDb(
            {
                "model_registry": FakeRegistryCollection(
                    [
                        make_entry(
                            "temp_prediction_1h_Prod",
                            doc_id="p1",
                            stage="production",
                            timestamp=dt(1),
                        ),
                        make_entry(
                            "temp_prediction_1h_New", doc_id="s1", stage="staging", timestamp=dt(5)
                        ),
                    ]
                )
            }
        )
        entry, is_fallback = inference.model_selection.select_entry(db, "temp_prediction_1h_")
        assert entry["_id"] == "p1"
        assert is_fallback is False

    def test_newest_staging_is_the_flagged_fallback(self):
        db = FakeDb(
            {
                "model_registry": FakeRegistryCollection(
                    [
                        make_entry(
                            "temp_prediction_1h_Old", doc_id="s0", stage="staging", timestamp=dt(2)
                        ),
                        make_entry(
                            "temp_prediction_1h_New", doc_id="s1", stage="staging", timestamp=dt(5)
                        ),
                    ]
                )
            }
        )
        entry, is_fallback = inference.model_selection.select_entry(db, "temp_prediction_1h_")
        assert entry["_id"] == "s1"
        assert is_fallback is True

    def test_nothing_matches(self):
        db = FakeDb({"model_registry": FakeRegistryCollection([])})
        assert inference.model_selection.select_entry(db, "non_existent_prefix") == (None, False)


class TestLoadLatestModel:
    def test_load_latest_model_gridfs_none_when_empty(self):
        mock_db = MagicMock()
        mock_fs = MagicMock()
        mock_db.__getitem__.return_value.find_one.return_value = None

        with patch.dict(os.environ, {}, clear=True):
            model, meta = inference.load_latest_model(
                mock_db, mock_fs, "temp_prediction_1h", MagicMock()
            )
            assert model is None
            assert meta is None

    def test_load_latest_model_mlflow_success_no_registry_entry(self):
        # MLflow artifact loads, but there is no model_registry doc -> fall back
        # to the MLflow-derived metadata (no interval, honest "MLflow-*" version).
        mock_db = MagicMock()
        mock_db["model_registry"].find_one.return_value = None
        mock_fs = MagicMock()
        mock_model = MagicMock()

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
            with patch("mlflow.spark.load_model", return_value=mock_model):
                model, meta = inference.load_latest_model(
                    mock_db, mock_fs, "temp_prediction_1h", MagicMock()
                )
                assert model == mock_model
                assert meta["version"].startswith("MLflow-")
                assert "interval" not in meta

    def test_load_latest_model_gridfs_wins_over_mlflow(self):
        # M6 regression: a registry doc WITH a gridfs_file_id is always served
        # from GridFS. MLflow is attempted only when the doc is absent or has no
        # file id — it must never win, and its metadata never replaces the
        # registry's.
        entry = {
            "model_name": "temp_prediction_1h_LinearRegression",
            "version": "20260925_201254",
            "stage": "production",
            "gridfs_file_id": "gridfs-123",
            "interval": {
                "level": 0.8,
                "lower_offset": -0.8971481336983871,
                "upper_offset": 0.5871447896789732,
            },
        }
        mock_db = MagicMock()
        mock_db["model_registry"].find_one.return_value = entry
        mock_model = MagicMock()

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
            with patch("mlflow.spark.load_model") as mlflow_load:
                with patch.object(inference.PipelineModel, "load", return_value=mock_model) as load:
                    with patch.object(inference, "_download_and_unzip", return_value="/tmp/x"):
                        model, meta = inference.load_latest_model(
                            mock_db, MagicMock(), "temp_prediction_1h", MagicMock()
                        )

        assert model == mock_model
        assert meta == entry
        assert meta["interval"]["lower_offset"] == pytest.approx(-0.8971481336983871)
        mlflow_load.assert_not_called()
        load.assert_called_once_with("/tmp/x")

    def test_load_latest_model_mlflow_only_without_gridfs_id(self):
        # No gridfs_file_id -> the ephemeral MLflow store is the only artifact
        # source left, but the registry doc still owns model_name/version.
        entry = {
            "model_name": "temp_prediction_1h_GBT",
            "version": "registry-version",
            "stage": "production",
        }
        mock_db = MagicMock()
        mock_db["model_registry"].find_one.return_value = entry
        mock_model = MagicMock()

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
            with patch("mlflow.spark.load_model", return_value=mock_model):
                model, meta = inference.load_latest_model(
                    mock_db, MagicMock(), "temp_prediction_1h", MagicMock()
                )

        assert model == mock_model
        assert meta == entry
        assert meta["version"] == "registry-version"  # never "MLflow-latest"

    def test_load_latest_model_warns_when_falling_back_to_staging(self, caplog):
        entry = {
            "model_name": "temp_prediction_1h_GBT",
            "version": "v1",
            "stage": "staging",
            "gridfs_file_id": "gridfs-123",
        }
        mock_db = MagicMock()
        # No production doc for this prefix: only the staging query matches.
        mock_db["model_registry"].find_one.side_effect = lambda query, **kwargs: (
            entry if query.get("stage") == "staging" else None
        )
        mock_model = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(inference.PipelineModel, "load", return_value=mock_model):
                with patch.object(inference, "_download_and_unzip", return_value="/tmp/x"):
                    with caplog.at_level(logging.WARNING, logger="inference"):
                        model, meta = inference.load_latest_model(
                            mock_db, MagicMock(), "temp_prediction_1h", MagicMock()
                        )

        assert model == mock_model
        assert meta is entry
        assert any("No production model" in r.message for r in caplog.records)


class TestLatestPerCity:
    def test_duplicate_feature_rows_collapse_to_one_row_per_city(self, spark):
        """A legacy duplicate must not become two predictions with the same _id."""
        schema = StructType(
            [
                StructField("city", StringType(), False),
                StructField("timestamp", TimestampType(), False),
                StructField("temperature", DoubleType(), True),
            ]
        )
        ts = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        df = spark.createDataFrame(
            [
                Row(city="Madrid", timestamp=ts, temperature=20.0),
                Row(city="Madrid", timestamp=ts, temperature=20.0),
                Row(city="Madrid", timestamp=datetime(2026, 10, 30, 6, 0, 0), temperature=18.0),
                Row(city="Palma", timestamp=ts, temperature=22.0),
            ],
            schema=schema,
        )

        out = inference.latest_per_city(df)

        assert out.count() == 2
        assert [row.city for row in out.orderBy("city").collect()] == ["Madrid", "Palma"]
        assert out.where("city = 'Madrid'").first().temperature == 20.0


class TestInferenceTransformation:
    @pytest.fixture()
    def pred_df(self, spark):
        schema = StructType(
            [
                StructField("city", StringType(), False),
                StructField("timestamp", TimestampType(), False),
                StructField("temperature", DoubleType(), True),
                StructField("predicted_temperature", DoubleType(), True),
                StructField("predicted_rain", DoubleType(), True),
            ]
        )
        data = [
            Row(
                city="Madrid",
                timestamp=datetime(2026, 6, 29, 12, 0, 0),
                temperature=20.0,
                predicted_temperature=21.5,
                predicted_rain=0.4,
            )
        ]
        return spark.createDataFrame(data, schema=schema)

    def test_build_output_df_exact_id(self, spark):
        # A tz-aware timestamp makes int(ts.timestamp()) the unambiguous UTC
        # epoch, matching F.unix_timestamp("timestamp") regardless of the
        # session/driver timezone.
        ts = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)
        schema = StructType(
            [
                StructField("city", StringType(), False),
                StructField("timestamp", TimestampType(), False),
                StructField("temperature", DoubleType(), True),
                StructField("predicted_temperature", DoubleType(), True),
                StructField("predicted_rain", DoubleType(), True),
            ]
        )
        df = spark.createDataFrame(
            [
                Row(
                    city="Madrid",
                    timestamp=ts,
                    temperature=20.0,
                    predicted_temperature=21.5,
                    predicted_rain=0.4,
                )
            ],
            schema=schema,
        )

        out = inference.build_output_df(
            df,
            temp_meta={"model_name": "t", "version": "v1"},
            rain_meta={"model_name": "r", "version": "v1"},
        )
        row = out.first()
        assert row._id == f"Madrid_{int(ts.timestamp())}_1h"


class TestOutputSchemaColumns:
    def test_output_schema_columns_exact_contract(self):
        expected = [
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
        assert inference.output_schema_columns() == expected


class TestBuildOutputDf:
    @pytest.fixture()
    def pred_df(self, spark):
        schema = StructType(
            [
                StructField("city", StringType(), False),
                StructField("timestamp", TimestampType(), False),
                StructField("temperature", DoubleType(), True),
                StructField("predicted_temperature", DoubleType(), True),
                StructField("predicted_rain", DoubleType(), True),
            ]
        )
        data = [
            Row(
                city="Madrid",
                timestamp=datetime(2026, 6, 29, 12, 0, 0),
                temperature=20.0,
                predicted_temperature=21.5,
                predicted_rain=0.4,
            )
        ]
        return spark.createDataFrame(data, schema=schema)

    def test_build_output_df_with_interval(self, spark, pred_df):
        temp_meta = {
            "model_name": "temp_prediction_1h_GBT",
            "version": "v1",
            "interval": {"level": 0.8, "lower_offset": -2.0, "upper_offset": 2.5},
        }
        rain_meta = {"model_name": "rain_prediction_1h_GBT", "version": "v1"}

        out = inference.build_output_df(pred_df, temp_meta, rain_meta)

        assert out.columns == inference.output_schema_columns()
        row = out.first()
        assert row.temp_lower == 21.5 + (-2.0)
        assert row.temp_upper == 21.5 + 2.5
        assert row.interval_level == 0.8
        # _id composition unchanged: {city}_{epoch(source_timestamp)}_{horizon}h
        assert row._id.startswith("Madrid_")
        assert row._id.endswith("_1h")

    def test_build_output_df_without_interval(self, spark, pred_df):
        out = inference.build_output_df(
            pred_df, temp_meta={"model_name": "legacy", "version": "v0"}, rain_meta={}
        )

        assert out.columns == inference.output_schema_columns()
        row = out.first()
        assert row.temp_lower is None
        assert row.temp_upper is None
        assert row.interval_level is None

    def test_build_output_df_partial_interval_treated_as_absent(self, spark, pred_df):
        # A malformed/partial interval block must not raise; it is nulled out.
        out = inference.build_output_df(
            pred_df,
            temp_meta={"model_name": "t", "version": "v1", "interval": {"level": 0.8}},
            rain_meta={},
        )

        row = out.first()
        assert row.temp_lower is None
        assert row.temp_upper is None
        assert row.interval_level is None

    def test_build_output_df_no_temp_meta(self, spark, pred_df):
        out = inference.build_output_df(pred_df, temp_meta=None, rain_meta=None)

        row = out.first()
        assert row.temp_lower is None
        assert row.temp_upper is None
        assert row.interval_level is None
        assert row.temp_model_name == "unknown"
        assert row.rain_model_name == "unknown"

    def test_build_output_df_registry_lineage_and_offsets(self, spark, pred_df):
        # The registry doc's model_name/version drive the row lineage, and its
        # interval offsets drive temp_lower/temp_upper (the production values
        # from the failing run).
        temp_meta = {
            "model_name": "temp_prediction_1h_LinearRegression",
            "version": "20260925_201254",
            "interval": {
                "level": 0.8,
                "lower_offset": -0.8971481336983871,
                "upper_offset": 0.5871447896789732,
            },
        }
        rain_meta = {
            "model_name": "rain_prediction_1h_GradientBoostedTrees",
            "version": "20260925_201257",
        }

        out = inference.build_output_df(pred_df, temp_meta, rain_meta)

        row = out.first()
        assert row.temp_model_name == "temp_prediction_1h_LinearRegression"
        assert row.temp_model_version == "20260925_201254"
        assert row.temp_lower == pytest.approx(21.5 + (-0.8971481336983871))
        assert row.temp_upper == pytest.approx(21.5 + 0.5871447896789732)
        assert row.interval_level == 0.8
        assert row.rain_model_name == "rain_prediction_1h_GradientBoostedTrees"

    def test_build_output_df_horizon_parameter(self, spark, pred_df):
        out = inference.build_output_df(
            pred_df,
            temp_meta={"model_name": "temp_prediction_6h_GBT", "version": "v1"},
            rain_meta={"model_name": "rain_prediction_6h_GBT", "version": "v1"},
            horizon=6,
        )
        row = out.first()
        assert row.horizon_hours == 6
        assert row._id.endswith("_6h")
        assert row.temp_model_name == "temp_prediction_6h_GBT"


class _FakeModel:
    """A pipeline stand-in whose transform adds a ``prediction`` column."""

    stages = []

    def __init__(self, offset):
        self.offset = offset

    def transform(self, df):
        return df.withColumn("prediction", df["temperature"] + self.offset)


class TestRunInferenceInterval:
    def test_warns_and_nulls_when_no_interval(self, spark, caplog):
        features = spark.createDataFrame(
            [
                Row(
                    city="Madrid",
                    timestamp=datetime(2026, 6, 29, 12, 0, 0),
                    temperature=20.0,
                )
            ],
            schema=StructType(
                [
                    StructField("city", StringType(), False),
                    StructField("timestamp", TimestampType(), False),
                    StructField("temperature", DoubleType(), True),
                ]
            ),
        )

        # Neither source yields an interval block -> warning + null columns.
        temp_meta = {"model_name": "legacy", "version": "v0"}
        rain_meta = {"model_name": "r", "version": "v1"}

        with caplog.at_level(logging.WARNING, logger="inference"):
            out = inference.run_inference(
                spark,
                _FakeModel(1.5),
                _FakeModel(0.0),
                features,
                "mongodb://fake",
                temp_meta,
                rain_meta,
                horizon=1,
            )

        assert any("no 'interval' block" in r.message for r in caplog.records)
        row = out.first()
        assert row.temp_lower is None
        assert row.temp_upper is None
        assert row.interval_level is None

    def test_run_inference_horizon_sets_id_and_horizon_hours(self, spark):
        features = spark.createDataFrame(
            [
                Row(
                    city="Madrid",
                    timestamp=datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc),
                    temperature=20.0,
                )
            ],
            schema=StructType(
                [
                    StructField("city", StringType(), False),
                    StructField("timestamp", TimestampType(), False),
                    StructField("temperature", DoubleType(), True),
                ]
            ),
        )
        out = inference.run_inference(
            spark,
            _FakeModel(1.5),
            _FakeModel(0.0),
            features,
            "mongodb://fake",
            {"model_name": "t", "version": "v1"},
            {"model_name": "r", "version": "v1"},
            horizon=6,
        )
        row = out.first()
        assert row.horizon_hours == 6
        assert row._id.endswith("_6h")

    def test_write_predictions_calls_save(self, spark):
        out = inference.build_output_df(
            spark.createDataFrame(
                [
                    Row(
                        city="Madrid",
                        timestamp=datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc),
                        temperature=20.0,
                        predicted_temperature=21.5,
                        predicted_rain=0.4,
                    )
                ],
                schema=StructType(
                    [
                        StructField("city", StringType(), False),
                        StructField("timestamp", TimestampType(), False),
                        StructField("temperature", DoubleType(), True),
                        StructField("predicted_temperature", DoubleType(), True),
                        StructField("predicted_rain", DoubleType(), True),
                    ]
                ),
            ),
            {"model_name": "t", "version": "v1"},
            {"model_name": "r", "version": "v1"},
            horizon=1,
        )
        with patch.object(DataFrameWriter, "save"):
            inference.write_predictions(out, "mongodb://fake")


def _latest_features_df(spark):
    """Two cities' latest feature rows (the shape ``load_latest_features`` returns)."""
    schema = StructType(
        [
            StructField("city", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("temperature", DoubleType(), True),
        ]
    )
    return spark.createDataFrame(
        [
            Row(city="Madrid", timestamp=datetime(2026, 6, 29, 12, 0, 0), temperature=20.0),
            Row(city="Valencia", timestamp=datetime(2026, 6, 29, 12, 0, 0), temperature=25.0),
        ],
        schema=schema,
    )


class TestMainMultiHorizon:
    """``main()`` loops the configured horizons, unions them into one write, and
    skips a horizon with no model without dropping the others (Contract 1)."""

    def _run_main(self, monkeypatch, spark, present_horizons):
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setattr(inference, "FEATURES_CONFIG", {"target_horizons": [1, 3]})
        monkeypatch.setattr(inference, "create_spark_session", lambda *a, **k: MagicMock())
        monkeypatch.setattr(inference, "MongoClient", lambda url: MagicMock())
        monkeypatch.setattr(inference, "GridFS", lambda db: MagicMock())
        features_df = _latest_features_df(spark)
        monkeypatch.setattr(inference, "load_latest_features", lambda s, m: features_df)

        def fake_load(_db, _fs, prefix, _spark):
            horizon = int(prefix.split("_")[-1].removesuffix("h"))
            if horizon not in present_horizons:
                return None, None
            kind = "temp" if prefix.startswith("temp") else "rain"
            meta = {"model_name": f"{kind}_prediction_{horizon}h_GBT", "version": f"v{horizon}"}
            return _FakeModel(1.0 if kind == "temp" else 0.0), meta

        monkeypatch.setattr(inference, "load_latest_model", fake_load)
        captured: dict = {}
        monkeypatch.setattr(
            inference, "write_predictions", lambda output, url: captured.update(output=output)
        )

        inference.main()
        return captured

    def test_unions_all_horizons_with_isolated_rows(self, spark, monkeypatch):
        captured = self._run_main(monkeypatch, spark, present_horizons={1, 3})

        output = captured["output"]
        assert output.count() == 4  # 2 cities x 2 horizons, in one write
        rows = output.orderBy("city", "horizon_hours").collect()
        assert sorted({row.horizon_hours for row in rows}) == [1, 3]
        # Every row carries its own horizon: id suffix and model lineage.
        for row in rows:
            assert row._id.startswith(f"{row.city}_")
            assert row._id.endswith(f"_{row.horizon_hours}h")
            assert row.temp_model_name == f"temp_prediction_{row.horizon_hours}h_GBT"
            assert row.rain_model_name == f"rain_prediction_{row.horizon_hours}h_GBT"
            assert row.interval_level is None  # no interval in the fake meta
        assert len({row._id for row in rows}) == 4

    def test_missing_model_horizon_is_skipped_not_fatal(self, spark, monkeypatch):
        captured = self._run_main(monkeypatch, spark, present_horizons={1})

        output = captured["output"]
        assert output.count() == 2  # horizon 3 skipped, horizon 1 still written
        assert {row.horizon_hours for row in output.collect()} == {1}


class TestLoadLatestFeaturesCache:
    """``load_latest_features`` materialises the joined frame into the cache so
    the horizon transforms and the write don't re-scan ``weather_features``."""

    def test_caches_before_count(self, spark):
        # ``spark`` is the session-scoped fixture: the real ``F.max("timestamp")``
        # in ``load_latest_features`` needs an active session, so this test must
        # not run on a bare MagicMock (self-contained under ``-k``/``--lf``).
        mock_spark = MagicMock()
        reader = MagicMock()
        reader.option.return_value = reader
        raw = MagicMock()
        raw.columns = []
        reader.load.return_value = raw
        mock_spark.read.format.return_value = reader

        latest = MagicMock()
        raw.groupBy.return_value.agg.return_value = latest

        frame = MagicMock()
        cached = MagicMock()
        frame.attach_mock(cached, "cached")
        latest.join.return_value.dropDuplicates.return_value = frame
        frame.cache.return_value = cached
        cached.count.return_value = 7

        result = inference.load_latest_features(mock_spark, "mongodb://fake")

        # ``.cache()`` returns a *distinct* frame, and ``load_latest_features``
        # must hand that exact object back — discarding the cached frame (and
        # silently re-scanning Atlas) would return ``frame`` instead.
        assert result is cached
        latest.join.return_value.dropDuplicates.assert_called_once_with(["city", "timestamp"])
        # ``.cache()`` must be called before ``.count()`` so the count itself
        # performs the materialisation that populates the cache.
        assert frame.method_calls == [call.cache(), call.cached.count()]


class Py4JJavaError(Exception):
    """Fake ``pyspark.errors.Py4JJavaError`` wrapping a Java Mongo socket fault.

    ``is_transient_mongo_error`` classifies on ``str(exc)`` text, not the class
    name, so this only needs to carry the transient driver message.
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class TestMainMongoRetry:
    """``main()`` retries transient Mongo faults on the load/write boundaries and
    unpersists the feature cache in ``finally``."""

    def _setup_main(self, monkeypatch, spark, features_df=None):
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setattr(inference, "FEATURES_CONFIG", {"target_horizons": [1]})
        monkeypatch.setattr(inference, "create_spark_session", lambda *a, **k: MagicMock())
        monkeypatch.setattr(inference, "MongoClient", lambda url: MagicMock())
        monkeypatch.setattr(inference, "GridFS", lambda db: MagicMock())

        if features_df is None:
            features_df = _latest_features_df(spark)
        monkeypatch.setattr(inference, "load_latest_features", lambda s, m: features_df)
        monkeypatch.setattr(
            inference,
            "load_latest_model",
            lambda _db, _fs, _prefix, _spark: (
                _FakeModel(1.0),
                {"model_name": "temp_prediction_1h_GBT", "version": "v1"},
            ),
        )
        return features_df

    def _inject_sleep(self, monkeypatch, sleeps):
        # ``run_with_mongo_retries`` captures ``time.sleep`` as a *default
        # argument*, so patching ``time.sleep`` would leave that default pointing
        # at the real sleep. Inject the recorder through the ``sleep`` parameter
        # instead, keeping the real retry/classification logic exercised.
        real_run = inference.run_with_mongo_retries
        monkeypatch.setattr(
            inference,
            "run_with_mongo_retries",
            partial(real_run, sleep=sleeps.append),
        )

    def test_write_predictions_retried_on_transient_save_error(self, spark, monkeypatch):
        self._setup_main(monkeypatch, spark)
        sleeps = []
        self._inject_sleep(monkeypatch, sleeps)

        save_calls = {"n": 0}

        def flaky_save(self):
            save_calls["n"] += 1
            if save_calls["n"] == 1:
                raise Py4JJavaError(
                    "com.mongodb.MongoSocketReadException: Prematurely reached end of stream"
                )
            return None

        with patch.object(DataFrameWriter, "save", flaky_save):
            inference.main()

        assert save_calls["n"] == 2  # first save failed, retry succeeded
        assert sleeps == [5.0]  # one backoff, no real delay

    def test_write_predictions_non_transient_not_retried(self, spark, monkeypatch):
        self._setup_main(monkeypatch, spark)
        sleeps = []
        self._inject_sleep(monkeypatch, sleeps)

        save_calls = {"n": 0}

        def failing_save(self):
            save_calls["n"] += 1
            raise Py4JJavaError(
                "org.apache.spark.SparkException: AnalysisException: cannot resolve column 'city'"
            )

        with patch.object(DataFrameWriter, "save", failing_save):
            with pytest.raises(Py4JJavaError):
                inference.main()

        assert save_calls["n"] == 1  # non-transient -> fail fast, no retry
        assert sleeps == []

    def test_main_unpersists_features_on_completion(self, spark, monkeypatch):
        features_df = MagicMock()
        features_df.count.return_value = 2
        self._setup_main(monkeypatch, spark, features_df=features_df)
        monkeypatch.setattr(inference, "write_predictions", lambda output, url: None)

        inference.main()

        features_df.unpersist.assert_called_once_with()

    def test_load_latest_features_retried_on_transient_error(self, spark, monkeypatch):
        self._setup_main(monkeypatch, spark)
        sleeps = []
        self._inject_sleep(monkeypatch, sleeps)

        load_calls = {"n": 0}
        features_df = _latest_features_df(spark)

        def flaky_load(_spark, _url):
            load_calls["n"] += 1
            if load_calls["n"] == 1:
                raise Py4JJavaError(
                    "com.mongodb.MongoSocketReadException: Prematurely reached end of stream"
                )
            return features_df

        monkeypatch.setattr(inference, "load_latest_features", flaky_load)
        monkeypatch.setattr(inference, "write_predictions", lambda output, url: None)

        inference.main()

        assert load_calls["n"] == 2  # transient error, then success
        assert sleeps == [5.0]  # one backoff, no real delay

    def test_load_latest_model_retried_on_transient_error(self, spark, monkeypatch):
        self._setup_main(monkeypatch, spark)
        sleeps = []
        self._inject_sleep(monkeypatch, sleeps)

        temp_loads = {"n": 0}

        def flaky_load(_db, _fs, prefix, _spark):
            kind = "temp" if prefix.startswith("temp") else "rain"
            if kind == "temp":
                temp_loads["n"] += 1
                if temp_loads["n"] == 1:
                    raise Py4JJavaError(
                        "com.mongodb.MongoSocketReadException: Prematurely reached end of stream"
                    )
            meta = {"model_name": f"{kind}_prediction_1h_GBT", "version": "v1"}
            return _FakeModel(1.0 if kind == "temp" else 0.0), meta

        monkeypatch.setattr(inference, "load_latest_model", flaky_load)
        captured = {}
        monkeypatch.setattr(
            inference, "write_predictions", lambda output, url: captured.update(output=output)
        )

        inference.main()

        assert temp_loads["n"] == 2  # temp model retried once, rain untouched
        assert sleeps == [5.0]
        assert captured["output"].count() == 2  # horizon loop still wrote its 2 rows

    def test_main_unpersists_features_when_no_rows(self, spark, monkeypatch):
        features_df = MagicMock()
        features_df.count.return_value = 0
        self._setup_main(monkeypatch, spark, features_df=features_df)

        inference.main()

        # Early return on an empty feature frame still unpersists in ``finally``.
        features_df.unpersist.assert_called_once_with()


class TestPredictionIdTimezoneIndependence:
    """``_id`` embeds ``epoch(source_timestamp)``; a session-TZ change must not
    mint a new key for the same prediction (that would re-insert rows)."""

    @staticmethod
    def _output(spark):
        ts = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)
        df = spark.createDataFrame(
            [
                Row(
                    city="Madrid",
                    timestamp=ts,
                    temperature=20.0,
                    predicted_temperature=21.0,
                    predicted_rain=0.1,
                )
            ],
            schema=StructType(
                [
                    StructField("city", StringType(), False),
                    StructField("timestamp", TimestampType(), False),
                    StructField("temperature", DoubleType(), True),
                    StructField("predicted_temperature", DoubleType(), True),
                    StructField("predicted_rain", DoubleType(), True),
                ]
            ),
        )
        return inference.build_output_df(
            df,
            temp_meta={"model_name": "t", "version": "v1"},
            rain_meta={"model_name": "r", "version": "v1"},
            horizon=1,
        )

    def test_same_id_under_non_utc_session_timezone(self, spark):
        expected = f"Madrid_{int(datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc).timestamp())}_1h"
        original = spark.conf.get("spark.sql.session.timeZone")
        try:
            spark.conf.set("spark.sql.session.timeZone", "UTC")
            utc_id = self._output(spark).first()._id
            spark.conf.set("spark.sql.session.timeZone", "Asia/Tokyo")
            tokyo_id = self._output(spark).first()._id
            spark.conf.set("spark.sql.session.timeZone", "America/New_York")
            ny_id = self._output(spark).first()._id
        finally:
            spark.conf.set("spark.sql.session.timeZone", original)

        assert utc_id == expected
        assert tokyo_id == expected
        assert ny_id == expected

    def test_create_spark_session_pins_utc(self):
        # Second belt: the real session builder must pin the zone too (read from
        # source — ``test_batch_processing`` stubs ``spark_config`` in
        # ``sys.modules``, so importing it here would yield a MagicMock).
        source = open(os.path.join(SPARK_CONFIG_DIR, "spark_config.py")).read()
        assert 'config("spark.sql.session.timeZone", "UTC")' in source


class _ScaleModel:
    """Pipeline stand-in: ``prediction = temperature * scale`` (probability-ish)."""

    stages = []

    def __init__(self, scale):
        self.scale = scale

    def transform(self, df):
        return df.withColumn("prediction", df["temperature"] * self.scale)


class TestShadowComparison:
    @pytest.fixture()
    def features(self, spark):
        return spark.createDataFrame(
            [
                Row(
                    city="Madrid",
                    timestamp=datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc),
                    temperature=20.0,
                ),
                Row(
                    city="Valencia",
                    timestamp=datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc),
                    temperature=25.0,
                ),
            ],
            schema=StructType(
                [
                    StructField("city", StringType(), False),
                    StructField("timestamp", TimestampType(), False),
                    StructField("temperature", DoubleType(), True),
                ]
            ),
        )

    def test_temperature_deltas(self, spark, features):
        metrics = inference.shadow_comparison(
            features, _FakeModel(1.0), _FakeModel(3.0), target="temperature"
        )
        assert metrics["n_samples"] == 2
        assert metrics["temp_abs_delta_mean"] == pytest.approx(2.0)
        assert metrics["temp_abs_delta_max"] == pytest.approx(2.0)
        assert metrics["last_source_timestamp"] == datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)

    def test_rain_disagreement_at_half(self, spark):
        df = spark.createDataFrame(
            [
                Row(
                    city="Madrid",
                    timestamp=datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc),
                    temperature=0.2,
                ),
                Row(
                    city="Valencia",
                    timestamp=datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc),
                    temperature=0.8,
                ),
            ],
            schema=StructType(
                [
                    StructField("city", StringType(), False),
                    StructField("timestamp", TimestampType(), False),
                    StructField("temperature", DoubleType(), True),
                ]
            ),
        )
        # production: 0.2 / 0.8 -> dry + wet; shadow: 0.5 / 2.0 -> wet + wet
        metrics = inference.shadow_comparison(df, _ScaleModel(1.0), _ScaleModel(2.5), target="rain")
        assert metrics["rain_disagreement_rate"] == pytest.approx(0.5)
        assert metrics["rain_prob_delta_mean"] == pytest.approx(0.75)
        assert "temp_abs_delta_mean" not in metrics


class TestShadowDoc:
    def test_temperature_payload_shape(self):
        prod = make_entry("temp_prediction_1h_Prod", doc_id="p1", stage="production", version="v1")
        shadow = make_entry("temp_prediction_1h_New", doc_id="s1", stage="staging", version="v2")
        metrics = {
            "n_samples": 15,
            "last_source_timestamp": dt(3),
            "temp_abs_delta_mean": 0.5,
            "temp_abs_delta_max": 1.5,
        }
        doc = inference.build_shadow_doc(prod, shadow, "temperature", 1, metrics, updated_at=dt(4))

        assert doc["_id"] == "p1_s1"
        assert doc["target"] == "temperature"
        assert doc["horizon_hours"] == 1
        assert doc["production_model_name"] == "temp_prediction_1h_Prod"
        assert doc["production_model_version"] == "v1"
        assert doc["shadow_model_name"] == "temp_prediction_1h_New"
        assert doc["shadow_model_version"] == "v2"
        assert doc["n_samples"] == 15
        assert doc["last_source_timestamp"] == dt(3)
        assert doc["temp_abs_delta_mean"] == 0.5
        assert doc["temp_abs_delta_max"] == 1.5
        assert doc["updated_at"] == dt(4)
        assert "rain_prob_delta_mean" not in doc

    def test_rain_payload_shape(self):
        prod = make_entry("rain_prediction_1h_Prod", doc_id="p1", stage="production")
        shadow = make_entry("rain_prediction_1h_New", doc_id="s1", stage="staging")
        metrics = {
            "n_samples": 10,
            "last_source_timestamp": dt(3),
            "rain_prob_delta_mean": 0.12,
            "rain_disagreement_rate": 0.1,
        }
        doc = inference.build_shadow_doc(prod, shadow, "rain", 1, metrics, updated_at=dt(4))

        assert doc["_id"] == "p1_s1"
        assert doc["target"] == "rain"
        assert doc["rain_prob_delta_mean"] == 0.12
        assert doc["rain_disagreement_rate"] == 0.1
        assert "temp_abs_delta_mean" not in doc

    def test_write_shadow_is_an_idempotent_upsert(self):
        db = FakeDb()
        doc = {"_id": "p1_s1", "n_samples": 1, "updated_at": dt(4)}
        inference.write_shadow(db, doc)
        inference.write_shadow(db, {**doc, "n_samples": 2})
        stored = db["model_shadow"].docs
        assert len(stored) == 1  # replace_one(upsert=True): never two rows
        assert stored[0]["n_samples"] == 2


class TestRunShadowAb:
    def _db(self):
        return FakeDb(
            {
                "model_registry": FakeRegistryCollection(
                    [
                        make_entry(
                            "temp_prediction_1h_Prod",
                            doc_id="p1",
                            stage="production",
                            timestamp=dt(1),
                        ),
                        # A challenger must be strictly newer than production.
                        make_entry(
                            "temp_prediction_1h_New",
                            doc_id="s1",
                            stage="staging",
                            timestamp=dt(2),
                        ),
                    ]
                )
            }
        )

    def test_skips_without_production(self, spark, monkeypatch):
        db = FakeDb(
            {
                "model_registry": FakeRegistryCollection(
                    [make_entry("temp_prediction_1h_New", doc_id="s1", stage="staging")]
                )
            }
        )
        monkeypatch.setattr(
            inference, "load_model_entry", MagicMock(side_effect=AssertionError("must not load"))
        )
        entry = db["model_registry"].docs[0]
        result = inference.run_shadow_ab(
            db,
            MagicMock(),
            "temp_prediction_1h_",
            _FakeModel(1.0),
            entry,
            _latest_features_df(spark),
            1,
            "temperature",
        )
        assert result is None
        assert db["model_shadow"].docs == []

    def test_writes_one_doc_per_pair(self, spark, monkeypatch):
        db = self._db()
        monkeypatch.setattr(inference, "load_model_entry", lambda fs, entry: _FakeModel(3.0))
        prod = db["model_registry"].find_one({"_id": "p1"})

        first = inference.run_shadow_ab(
            db,
            MagicMock(),
            "temp_prediction_1h_",
            _FakeModel(1.0),
            prod,
            _latest_features_df(spark),
            1,
            "temperature",
        )
        second = inference.run_shadow_ab(
            db,
            MagicMock(),
            "temp_prediction_1h_",
            _FakeModel(1.0),
            prod,
            _latest_features_df(spark),
            1,
            "temperature",
        )

        assert first["_id"] == "p1_s1"
        assert second["_id"] == "p1_s1"
        docs = db["model_shadow"].docs
        assert len(docs) == 1  # idempotent: one doc per (production, shadow) pair
        assert docs[0]["n_samples"] == 2
        assert docs[0]["temp_abs_delta_max"] == pytest.approx(2.0)

    def test_shadow_failure_is_logged_and_skipped(self, spark, monkeypatch, caplog):
        db = self._db()
        prod = db["model_registry"].find_one({"_id": "p1"})

        def boom(_fs, _entry):
            raise RuntimeError("gridfs unavailable")

        monkeypatch.setattr(inference, "load_model_entry", boom)

        with caplog.at_level(logging.WARNING, logger="inference"):
            result = inference.run_shadow_ab(
                db,
                MagicMock(),
                "temp_prediction_1h_",
                _FakeModel(1.0),
                prod,
                _latest_features_df(spark),
                1,
                "temperature",
            )

        assert result is None  # never raises into inference
        assert db["model_shadow"].docs == []
        assert any("Shadow A/B" in r.message for r in caplog.records)


class TestMainShadowWiring:
    """``main()`` runs the A/B only for a real production model, never on the
    staging fallback, and a shadow fault cannot fail the run."""

    def _run_main(self, monkeypatch, spark, stage):
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setattr(inference, "FEATURES_CONFIG", {"target_horizons": [1]})
        monkeypatch.setattr(inference, "create_spark_session", lambda *a, **k: MagicMock())
        monkeypatch.setattr(inference, "MongoClient", lambda url: MagicMock())
        monkeypatch.setattr(inference, "GridFS", lambda db: MagicMock())
        features_df = _latest_features_df(spark)
        monkeypatch.setattr(inference, "load_latest_features", lambda s, m: features_df)

        def fake_load(_db, _fs, prefix, _spark):
            kind = "temp" if prefix.startswith("temp") else "rain"
            return _FakeModel(1.0), {
                "_id": f"{kind}-prod",
                "model_name": f"{kind}_prediction_1h_GBT",
                "version": "v1",
                "stage": stage,
            }

        monkeypatch.setattr(inference, "load_latest_model", fake_load)
        writes: list = []
        monkeypatch.setattr(
            inference, "write_predictions", lambda output, url: writes.append(output)
        )
        calls: list = []
        monkeypatch.setattr(
            inference,
            "run_shadow_ab",
            lambda *args: calls.append({"prefix": args[2], "target": args[7]}),
        )
        inference.main()
        return calls, writes

    def test_shadow_runs_for_production_models(self, spark, monkeypatch):
        calls, writes = self._run_main(monkeypatch, spark, "production")
        assert writes, "predictions must still be written"
        assert calls == [
            {"prefix": "temp_prediction_1h", "target": "temperature"},
            {"prefix": "rain_prediction_1h", "target": "rain"},
        ]

    def test_shadow_never_runs_on_the_staging_fallback(self, spark, monkeypatch):
        calls, writes = self._run_main(monkeypatch, spark, "staging")
        assert writes
        assert calls == []


# ---------------------------------------------------------------------------
# Prediction coverage (the 2026-10-30 "scored 1 of 15 cities" incident)
# ---------------------------------------------------------------------------


def _coverage_output(spark, cities, oldest=None):
    """A prediction-shaped frame with one row per city (source_timestamp set)."""
    schema = StructType(
        [
            StructField("city", StringType(), False),
            StructField("source_timestamp", TimestampType(), False),
            StructField("horizon_hours", DoubleType(), True),
        ]
    )
    newest = datetime(2026, 10, 30, 9, 0, 0)
    rows = [
        Row(
            city=city,
            source_timestamp=(oldest if (oldest is not None and index == 0) else newest),
            horizon_hours=1.0,
        )
        for index, city in enumerate(cities)
    ]
    return spark.createDataFrame(rows, schema=schema)


def _features_for_cities(spark, cities):
    schema = StructType(
        [
            StructField("city", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("temperature", DoubleType(), True),
        ]
    )
    return spark.createDataFrame(
        [
            Row(city=city, timestamp=datetime(2026, 10, 30, 9, 0, 0), temperature=20.0)
            for city in cities
        ],
        schema=schema,
    )


class TestCoverageWarningThreshold:
    """Pure threshold logic: warn under 90 % coverage or >12 h of staleness."""

    def test_no_warning_at_full_coverage(self):
        stats = {"scored_cities": 14, "available_cities": 14, "oldest_behind_hours": 0.0}
        assert inference.coverage_warning(stats) is None

    def test_warns_when_one_of_fourteen_cities_is_scored(self):
        stats = {"scored_cities": 1, "available_cities": 14, "oldest_behind_hours": 0.0}
        warning = inference.coverage_warning(stats)
        assert warning is not None
        assert "scored 1/14 cities" in warning

    def test_boundary_is_ninety_percent(self):
        exact = {"scored_cities": 9, "available_cities": 10, "oldest_behind_hours": 0.0}
        below = {"scored_cities": 8, "available_cities": 10, "oldest_behind_hours": 0.0}
        assert inference.coverage_warning(exact) is None
        assert inference.coverage_warning(below) is not None

    def test_warns_on_a_stale_scored_city(self):
        stats = {"scored_cities": 14, "available_cities": 14, "oldest_behind_hours": 12.5}
        warning = inference.coverage_warning(stats)
        assert warning is not None
        assert "12.5 h behind" in warning

    def test_no_warning_at_the_staleness_boundary(self):
        stats = {"scored_cities": 14, "available_cities": 14, "oldest_behind_hours": 12.0}
        assert inference.coverage_warning(stats) is None

    def test_constants_are_the_frozen_contract(self):
        assert inference.COVERAGE_WARN_RATIO == 0.9
        assert inference.STALE_SOURCE_HOURS == 12.0


class TestReportPredictionCoverage:
    """``report_prediction_coverage`` logs, annotates, and never raises."""

    def test_warns_and_annotates_when_only_one_city_is_scorable(self, spark, caplog, capsys):
        output = _coverage_output(spark, ["Pasadena"])
        features = _features_for_cities(spark, [f"City{index}" for index in range(14)])

        with caplog.at_level(logging.INFO, logger=inference.logger.name):
            warning = inference.report_prediction_coverage(output, features)

        assert warning is not None
        messages = [record.getMessage() for record in caplog.records]
        assert any("Scored 1/14 cities" in message for message in messages)
        assert any(
            record.levelno == logging.WARNING and "Prediction coverage" in record.getMessage()
            for record in caplog.records
        )
        err = capsys.readouterr().err
        assert "::warning title=Prediction coverage::scored 1/14 cities" in err

    def test_no_warning_or_annotation_at_full_coverage(self, spark, caplog, capsys):
        cities = [f"City{index}" for index in range(14)]
        output = _coverage_output(spark, cities)
        features = _features_for_cities(spark, cities)

        with caplog.at_level(logging.INFO, logger=inference.logger.name):
            warning = inference.report_prediction_coverage(output, features)

        assert warning is None
        assert any("Scored 14/14 cities" in record.getMessage() for record in caplog.records)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert "::warning" not in capsys.readouterr().err

    def test_annotation_flags_a_stale_city(self, spark, capsys):
        cities = ["Pasadena", "Madrid"]
        output = _coverage_output(spark, cities, oldest=datetime(2026, 9, 25, 9, 0, 0))
        features = _features_for_cities(spark, cities)

        warning = inference.report_prediction_coverage(output, features)

        assert warning is not None
        assert "behind the newest" in warning
        assert "::warning title=Prediction coverage::" in capsys.readouterr().err

    def test_a_broken_check_never_blocks_the_write(self, spark, capsys):
        # A frame without ``city`` makes the counts blow up; the diagnostic must
        # swallow it, print no annotation, and let inference keep going.
        broken = spark.createDataFrame(
            [(datetime(2026, 10, 30, 9, 0, 0),)],
            schema=StructType([StructField("source_timestamp", TimestampType(), True)]),
        )
        features = _features_for_cities(spark, ["Madrid"])

        assert inference.report_prediction_coverage(broken, features) is None
        assert "::warning" not in capsys.readouterr().err

    def test_main_reports_coverage_before_writing(self, spark, monkeypatch, caplog, capsys):
        """``main`` runs the check after the union and before the write, and the
        write still happens when coverage is short (never raise)."""
        monkeypatch.setenv("MONGO_URI", "mongodb://fake")
        monkeypatch.setattr(inference, "FEATURES_CONFIG", {"target_horizons": [1]})
        monkeypatch.setattr(inference, "create_spark_session", lambda *a, **k: MagicMock())
        monkeypatch.setattr(inference, "MongoClient", lambda url: MagicMock())
        monkeypatch.setattr(inference, "GridFS", lambda db: MagicMock())
        holder = MagicMock()
        holder.count.return_value = 1
        holder.unpersist = MagicMock()
        # The cached feature frame's distinct-city count (the coverage denominator).
        holder.select.return_value.distinct.return_value.count.return_value = 1
        monkeypatch.setattr(inference, "load_latest_features", lambda s, m: holder)
        monkeypatch.setattr(
            inference,
            "load_latest_model",
            lambda _db, _fs, _prefix, _spark: (
                _FakeModel(1.0),
                {"model_name": "temp_prediction_1h_GBT", "version": "v1"},
            ),
        )
        output = _coverage_output(spark, ["Pasadena"])
        monkeypatch.setattr(inference, "run_inference", lambda *a, **k: output)
        written = {}
        monkeypatch.setattr(
            inference, "write_predictions", lambda out, url: written.update(rows=out.count())
        )

        with caplog.at_level(logging.INFO, logger=inference.logger.name):
            inference.main()

        assert written == {"rows": 1}, "the write must still happen"
        assert any("Scored 1/1 cities" in record.getMessage() for record in caplog.records)
        # 1/1 is full coverage: no annotation, nothing raised.
        assert "::warning" not in capsys.readouterr().err
