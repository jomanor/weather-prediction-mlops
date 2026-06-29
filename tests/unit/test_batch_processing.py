"""
tests/unit/test_batch_processing.py
=====================================
Unit tests for spark/spark-jobs/batch_processing.py

Uses a local SparkSession (no cluster, no MongoDB).
All DB read/writes are stubbed out.
"""

import sys
import os
import unittest.mock as mock
import pytest

from pyspark.sql import Row
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
)
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — batch_processing.py imports spark_config from /opt/config
# We mock that import so we can run offline.
# ---------------------------------------------------------------------------
SPARK_JOBS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "spark", "spark-jobs"
)
sys.path.insert(0, SPARK_JOBS_DIR)

# Stub spark_config before importing batch_processing
fake_config = mock.MagicMock()
fake_config.FEATURES_CONFIG = {
    "lag_periods": [1, 3],
    "window_sizes": [6],
    "target_horizon": 1,
}
fake_config.create_spark_session = mock.MagicMock()
sys.modules.setdefault("spark_config", fake_config)

import batch_processing as bp  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_df(spark):
    """A small DataFrame that mimics the output of extract_weather_data()."""
    schema = StructType(
        [
            StructField("city", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("feels_like", DoubleType(), True),
            StructField("humidity", DoubleType(), True),
            StructField("pressure", DoubleType(), True),
            StructField("pressure_msl", DoubleType(), True),
            StructField("dew_point", DoubleType(), True),
            StructField("wind_speed", DoubleType(), True),
            StructField("wind_direction", DoubleType(), True),
            StructField("wind_gusts", DoubleType(), True),
            StructField("wind_speed_80m", DoubleType(), True),
            StructField("wind_direction_80m", DoubleType(), True),
            StructField("cloud_cover", DoubleType(), True),
            StructField("precipitation", DoubleType(), True),
            StructField("rain", DoubleType(), True),
            StructField("snowfall", DoubleType(), True),
            StructField("visibility", DoubleType(), True),
            StructField("shortwave_radiation", DoubleType(), True),
            StructField("cape", DoubleType(), True),
            StructField("weather_code", DoubleType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
        ]
    )

    rows = [
        Row(
            city="Madrid",
            timestamp=datetime(2025, 6, 15, 10 + i, 0, 0),
            temperature=20.0 + i,
            feels_like=19.0 + i,
            humidity=55.0,
            pressure=1013.0,
            pressure_msl=1015.0,
            dew_point=13.5,
            wind_speed=5.2,
            wind_direction=270.0,
            wind_gusts=8.1,
            wind_speed_80m=7.3,
            wind_direction_80m=265.0,
            cloud_cover=20.0,
            precipitation=0.0,
            rain=0.0,
            snowfall=0.0,
            visibility=24140.0,
            shortwave_radiation=350.0,
            cape=0.0,
            weather_code=1.0,
            latitude=40.4168,
            longitude=-3.7038,
        )
        for i in range(5)
    ]
    return spark.createDataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# add_atmospheric_features
# ---------------------------------------------------------------------------


class TestAddAtmosphericFeatures:
    def test_specific_humidity_column_added(self, sample_df):
        result = bp.add_atmospheric_features(sample_df)
        assert "specific_humidity" in result.columns

    def test_wind_u_wind_v_columns_added(self, sample_df):
        result = bp.add_atmospheric_features(sample_df)
        assert "wind_u" in result.columns
        assert "wind_v" in result.columns

    def test_specific_humidity_no_nulls_for_valid_input(self, sample_df):
        result = bp.add_atmospheric_features(sample_df)
        null_count = result.filter(result["specific_humidity"].isNull()).count()
        assert (
            null_count == 0
        ), "specific_humidity should not be null when inputs are valid"

    def test_specific_humidity_positive(self, sample_df):
        """Specific humidity is a ratio (kg/kg) and must be positive."""
        result = bp.add_atmospheric_features(sample_df)
        negative_count = result.filter(result["specific_humidity"] < 0).count()
        assert negative_count == 0

    def test_wind_components_westerly_direction(self, sample_df):
        """For 270° (westerly) wind: u should be positive, v near zero."""
        result = bp.add_atmospheric_features(sample_df)
        row = result.select("wind_u", "wind_v").first()
        # 270° → u = -speed * sin(270°) = -speed * (-1) = +speed
        assert row["wind_u"] > 0, "Westerly wind should have positive u component"
        assert (
            abs(row["wind_v"]) < 0.5
        ), "Westerly wind should have v component near zero"


# ---------------------------------------------------------------------------
# create_time_features
# ---------------------------------------------------------------------------


class TestCreateTimeFeatures:
    def test_hour_column(self, sample_df):
        result = bp.create_time_features(sample_df)
        assert "hour" in result.columns

    def test_cyclical_columns(self, sample_df):
        result = bp.create_time_features(sample_df)
        for col in ("hour_sin", "hour_cos", "month_sin", "month_cos"):
            assert col in result.columns

    def test_cyclical_values_in_range(self, sample_df):
        result = bp.create_time_features(sample_df)
        row = result.select("hour_sin", "hour_cos").first()
        assert -1.0 <= row["hour_sin"] <= 1.0
        assert -1.0 <= row["hour_cos"] <= 1.0


# ---------------------------------------------------------------------------
# create_target_variable
# ---------------------------------------------------------------------------


class TestCreateTargetVariable:
    def test_target_column_added(self, sample_df):
        result = bp.create_target_variable(sample_df, horizon=1)
        assert "target_temp_1h" in result.columns
        assert "target_will_rain_1h" in result.columns

    def test_last_row_target_is_null(self, sample_df):
        """The last row has no future value, so target must be null."""
        result = bp.create_target_variable(sample_df, horizon=1)
        # Sort by timestamp descending so first row of this query is the last one
        from pyspark.sql import functions as F

        null_count = result.filter(F.col("target_temp_1h").isNull()).count()
        assert (
            null_count >= 1
        ), "At least one row should have null target (no future data)"

    def test_rain_target_is_binary(self, sample_df):
        """target_will_rain must be 0 or 1 (or null for the last row)."""
        from pyspark.sql import functions as F

        result = bp.create_target_variable(sample_df, horizon=1)
        non_binary = result.filter(
            F.col("target_will_rain_1h").isNotNull()
            & ~F.col("target_will_rain_1h").isin([0, 1])
        ).count()
        assert non_binary == 0


# ---------------------------------------------------------------------------
# create_precipitation_rolling_sum
# ---------------------------------------------------------------------------


class TestPrecipRollingSum:
    def test_precip_sum_column_added(self, sample_df):
        result = bp.create_precipitation_rolling_sum(sample_df)
        assert "precip_sum_6h" in result.columns

    def test_precip_sum_nonnegative(self, sample_df):
        from pyspark.sql import functions as F

        result = bp.create_precipitation_rolling_sum(sample_df)
        neg_count = result.filter(F.col("precip_sum_6h") < 0).count()
        assert neg_count == 0
