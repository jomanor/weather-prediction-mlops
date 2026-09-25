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
from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql import DataFrameWriter, Row
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Ensure spark jobs and config are on sys.path
SPARK_JOBS_DIR = os.path.join(os.path.dirname(__file__), "../../spark/spark-jobs")
SPARK_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../spark/config")
for d in (SPARK_JOBS_DIR, SPARK_CONFIG_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import inference  # noqa: E402


class TestModelRegistryLookup:
    def test_latest_model_entry_found(self):
        mock_db = MagicMock()
        mock_db.__getitem__.return_value.find_one.return_value = {
            "model_name": "temp_prediction_1h_GBT",
            "gridfs_file_id": "mock_id_123",
            "version": "20260629_120000",
        }

        entry = inference._latest_model_entry(mock_db, "temp_prediction_1h")
        assert entry is not None
        assert entry["model_name"] == "temp_prediction_1h_GBT"
        mock_db.__getitem__.assert_called_with("model_registry")

    def test_latest_model_entry_not_found(self):
        mock_db = MagicMock()
        mock_db.__getitem__.return_value.find_one.return_value = None

        entry = inference._latest_model_entry(mock_db, "non_existent_prefix")
        assert entry is None


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

    def test_load_latest_model_mlflow_artifact_uses_registry_metadata(self):
        # Regression: MLflow artifact wins, but metadata must come from the
        # newest model_registry doc so interval + lineage are not dropped.
        entry = {
            "model_name": "temp_prediction_1h_LinearRegression",
            "version": "20260925_201254",
            "gridfs_file_id": "mock_id",
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
            with patch("mlflow.spark.load_model", return_value=mock_model):
                model, meta = inference.load_latest_model(
                    mock_db, MagicMock(), "temp_prediction_1h", MagicMock()
                )

        assert model == mock_model
        assert meta == entry
        assert meta["model_name"] == "temp_prediction_1h_LinearRegression"
        assert meta["version"] == "20260925_201254"
        assert meta["interval"]["lower_offset"] == pytest.approx(-0.8971481336983871)


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
            with patch.object(DataFrameWriter, "save"):
                inference.run_inference(
                    spark,
                    _FakeModel(1.5),
                    _FakeModel(0.0),
                    features,
                    "mongodb://fake",
                    temp_meta,
                    rain_meta,
                )

        assert any("no 'interval' block" in r.message for r in caplog.records)
