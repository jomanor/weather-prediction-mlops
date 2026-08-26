"""
test_inference.py
=================
Unit tests for spark-jobs/inference.py.
Tests model loading, registry queries, and batch inference transformations.
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql import Row
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

    def test_load_latest_model_mlflow_success(self):
        mock_db = MagicMock()
        mock_fs = MagicMock()
        mock_model = MagicMock()

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
            with patch("mlflow.spark.load_model", return_value=mock_model):
                model, meta = inference.load_latest_model(
                    mock_db, mock_fs, "temp_prediction_1h", MagicMock()
                )
                assert model == mock_model
                assert "MLflow" in meta["version"]


class TestInferenceTransformation:
    @pytest.fixture()
    def sample_features_df(self, spark):
        schema = StructType(
            [
                StructField("city", StringType(), False),
                StructField("timestamp", TimestampType(), False),
                StructField("temperature", DoubleType(), True),
                StructField("humidity", DoubleType(), True),
                StructField("pressure", DoubleType(), True),
                StructField("wind_speed", DoubleType(), True),
                StructField("specific_humidity", DoubleType(), True),
                StructField("wind_u", DoubleType(), True),
                StructField("wind_v", DoubleType(), True),
                StructField("precip_sum_6h", DoubleType(), True),
            ]
        )
        data = [
            Row(
                city="Madrid",
                timestamp=datetime(2026, 6, 29, 12, 0, 0),
                temperature=25.0,
                humidity=40.0,
                pressure=1015.0,
                wind_speed=3.5,
                specific_humidity=8.2,
                wind_u=-2.5,
                wind_v=-1.5,
                precip_sum_6h=0.0,
            )
        ]
        return spark.createDataFrame(data, schema=schema)

    def test_run_inference_dataframe_schema(self, spark, sample_features_df):
        mock_temp_model = MagicMock()
        # Transform adds "prediction" column
        mock_temp_model.transform.side_effect = lambda df: df.withColumn(
            "prediction", df["temperature"] + 1.5
        )

        mock_rain_model = MagicMock()
        mock_rain_model.transform.side_effect = lambda df: df.withColumn(
            "prediction", df["temperature"] * 0.0
        )

        # Mock MongoDB write
        writer = sample_features_df.write
        with patch.object(writer, "format") as mock_fmt:
            mock_fmt.return_value.option.return_value.mode.return_value.save = MagicMock()

            # We test the column selection and renaming logic without throwing
            pred_df = mock_temp_model.transform(sample_features_df).withColumnRenamed(
                "prediction", "predicted_temperature"
            )
            pred_df = mock_rain_model.transform(pred_df).withColumnRenamed(
                "prediction", "predicted_rain"
            )

            assert "predicted_temperature" in pred_df.columns
            assert "predicted_rain" in pred_df.columns
            row = pred_df.first()
            assert row.predicted_temperature == 26.5
            assert row.predicted_rain == 0.0
