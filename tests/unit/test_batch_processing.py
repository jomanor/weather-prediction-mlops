"""
tests/unit/test_batch_processing.py
=====================================
Unit tests for spark/spark-jobs/batch_processing.py

Uses a local SparkSession (no cluster, no MongoDB).
All DB read/writes are stubbed out.
"""

import ast
import inspect
import os
import sys
import unittest.mock as mock
from datetime import datetime, timedelta, timezone

import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Path setup — batch_processing.py imports spark_config from /opt/config
# We mock that import so we can run offline.
# ---------------------------------------------------------------------------
SPARK_JOBS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "spark", "spark-jobs")
sys.path.insert(0, SPARK_JOBS_DIR)

# Stub spark_config before importing batch_processing
fake_config = mock.MagicMock()
fake_config.FEATURES_CONFIG = {
    "lag_periods": [1, 3],
    "window_sizes": [6],
    "target_horizons": [1, 3],
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
        assert null_count == 0, "specific_humidity should not be null when inputs are valid"

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
        assert abs(row["wind_v"]) < 0.5, "Westerly wind should have v component near zero"


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
# create_time_features — local hour (not UTC)
# ---------------------------------------------------------------------------


class TestLocalHour:
    """Diurnal features must use local time, not the UTC timestamp."""

    def test_solar_offset_seconds_rounds_to_nearest_hour(self):
        # Madrid (-3.70°) → 0h; Pasadena (-118.15°) → -8h; +30° → +2h.
        assert bp.solar_offset_seconds(-3.7038) == 0
        assert bp.solar_offset_seconds(-118.15) == -8 * 3600
        assert bp.solar_offset_seconds(2.1734) == 0
        assert bp.solar_offset_seconds(30.0) == 2 * 3600

    def test_local_hour_uses_utc_offset_when_present(self, spark):
        """Same UTC instant, offset +1h vs -7h => local hours 8 apart (mod 24)."""
        schema = StructType(
            [
                StructField("city", StringType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("longitude", DoubleType(), True),
                StructField("utc_offset_seconds", LongType(), True),
            ]
        )
        base = 1_750_000_000
        ts = datetime.fromtimestamp(base, tz=timezone.utc)
        rows = [
            Row(city="Madrid", timestamp=ts, longitude=-3.7038, utc_offset_seconds=3600),
            Row(city="Pasadena", timestamp=ts, longitude=-118.15, utc_offset_seconds=-25200),
        ]
        df = spark.createDataFrame(rows, schema=schema)
        result = bp.create_time_features(df)

        hours = {r["city"]: r["hour"] for r in result.select("city", "hour").collect()}
        # CET (+1) vs PST (-7): 8 hours apart regardless of the session timezone.
        assert (hours["Madrid"] - hours["Pasadena"]) % 24 == 8

    def test_local_hour_falls_back_to_longitude(self, spark):
        """Without utc_offset_seconds, the longitude-based solar offset applies."""
        schema = StructType(
            [
                StructField("city", StringType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("longitude", DoubleType(), True),
                StructField("utc_offset_seconds", LongType(), True),
            ]
        )
        base = 1_750_000_000
        ts = datetime.fromtimestamp(base, tz=timezone.utc)
        rows = [
            Row(
                city="Pasadena_explicit", timestamp=ts, longitude=-118.15, utc_offset_seconds=-28800
            ),
            Row(city="Pasadena_fallback", timestamp=ts, longitude=-118.15, utc_offset_seconds=None),
        ]
        df = spark.createDataFrame(rows, schema=schema)
        result = bp.create_time_features(df)

        hours = {r["city"]: r["hour"] for r in result.select("city", "hour").collect()}
        # Solar fallback for -118.15° is -8h: must match the explicit offset.
        assert hours["Pasadena_explicit"] == hours["Pasadena_fallback"]
        assert hours["Pasadena_fallback"] is not None


# ---------------------------------------------------------------------------
# create_target_variable
# ---------------------------------------------------------------------------


class TestCreateTargetVariable:
    def test_target_column_added(self, sample_df):
        result = bp.create_target_variable(sample_df, horizons=[1])
        assert "target_temp_1h" in result.columns
        assert "target_will_rain_1h" in result.columns

    def test_multiple_horizons_written_in_one_pass(self, sample_df):
        result = bp.create_target_variable(sample_df, horizons=[1, 3, 6])
        for h in (1, 3, 6):
            assert f"target_temp_{h}h" in result.columns
            assert f"target_will_rain_{h}h" in result.columns

    def test_last_row_target_is_null(self, sample_df):
        """The last row has no future value, so target must be null."""
        result = bp.create_target_variable(sample_df, horizons=[1])
        # Sort by timestamp descending so first row of this query is the last one
        from pyspark.sql import functions as F

        null_count = result.filter(F.col("target_temp_1h").isNull()).count()
        assert null_count >= 1, "At least one row should have null target (no future data)"

    def test_rain_target_is_binary(self, sample_df):
        """target_will_rain must be 0 or 1 (or null for the last row)."""
        from pyspark.sql import functions as F

        result = bp.create_target_variable(sample_df, horizons=[1])
        non_binary = result.filter(
            F.col("target_will_rain_1h").isNotNull() & ~F.col("target_will_rain_1h").isin([0, 1])
        ).count()
        assert non_binary == 0


# ---------------------------------------------------------------------------
# save_features_to_mongodb — freshness fix (dropna subset = observation inputs)
# ---------------------------------------------------------------------------


class TestSaveFeaturesDropnaSubset:
    def test_observation_inputs_exclude_targets(self):
        # The dropna subset is the observation inputs only; a target column in
        # the subset would discard the newest observation row (null lead target)
        # and force every healthy rebuild to land ~6 h stale.
        assert bp.OBSERVATION_INPUT_COLUMNS == [
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
        ]
        assert not any(c.startswith("target_") for c in bp.OBSERVATION_INPUT_COLUMNS)


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


# ---------------------------------------------------------------------------
# Upsert _id, prune cutoff, transient retries — pure, no Spark/Mongo
# ---------------------------------------------------------------------------


class _TransientError(Exception):
    """Stand-in for a pymongo OperationFailure carrying an error code."""

    def __init__(self, code=None, details=None):
        super().__init__(f"code={code}")
        if code is not None:
            self.code = code
        if details is not None:
            self.details = details


class MongoNodeIsRecoveringException(Exception):
    pass


class MongoConnectionPoolClearedException(Exception):
    pass


class TestFeatureDocId:
    """``_id = {city}_{epoch_seconds(timestamp)}`` — the working convention."""

    def test_id_is_city_underscore_epoch_seconds(self):
        assert bp.feature_doc_id("Madrid", 1_750_000_000) == "Madrid_1750000000"

    def test_id_matches_prediction_convention(self):
        # Same base composition as inference's _id, minus the horizon suffix.
        assert bp.feature_doc_id("A Coruña", 1_750_003_600) == "A Coruña_1750003600"

    def test_id_truncates_sub_second_timestamps(self):
        assert bp.feature_doc_id("Madrid", 1_750_000_000.987) == "Madrid_1750000000"

    def test_id_is_stable_and_distinct(self):
        assert bp.feature_doc_id("Madrid", 1_750_000_000) == bp.feature_doc_id(
            "Madrid", 1_750_000_000
        )
        assert bp.feature_doc_id("Madrid", 1_750_000_000) != bp.feature_doc_id(
            "Madrid", 1_750_000_001
        )
        assert bp.feature_doc_id("Madrid", 1_750_000_000) != bp.feature_doc_id(
            "Valencia", 1_750_000_000
        )


class TestPruneCutoff:
    def test_default_retention_matches_raw_weather_ttl(self):
        assert bp.RETENTION_DAYS == 180

    def test_cutoff_is_now_minus_retention(self):
        now = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
        assert bp.prune_cutoff(now) == now - timedelta(days=180)

    def test_cutoff_honours_explicit_retention(self):
        now = datetime(2026, 9, 28, tzinfo=timezone.utc)
        assert bp.prune_cutoff(now, retention_days=30) == now - timedelta(days=30)

    def test_prune_filter_is_a_bounded_range_predicate(self):
        cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc)
        assert bp.prune_filter(cutoff) == {"timestamp": {"$lt": cutoff}}


class TestTransientMongoErrors:
    def test_code_91_shutdown_in_progress_is_transient(self):
        assert bp.is_transient_mongo_error(_TransientError(code=91)) is True

    def test_code_116_is_transient(self):
        assert bp.is_transient_mongo_error(_TransientError(code=116)) is True

    def test_code_from_details_is_transient(self):
        assert bp.is_transient_mongo_error(_TransientError(details={"code": 91})) is True

    def test_node_is_recovering_exception_is_transient(self):
        assert bp.is_transient_mongo_error(MongoNodeIsRecoveringException("x")) is True

    def test_connection_pool_cleared_exception_is_transient(self):
        assert bp.is_transient_mongo_error(MongoConnectionPoolClearedException("x")) is True

    def test_other_errors_are_not_transient(self):
        assert bp.is_transient_mongo_error(ValueError("bad schema")) is False
        assert bp.is_transient_mongo_error(_TransientError(code=11000)) is False


class TestRetryBackoff:
    def test_delay_is_exponential(self):
        assert bp.retry_delay_seconds(0, base=5.0, factor=2.0, cap=60.0) == 5.0
        assert bp.retry_delay_seconds(1, base=5.0, factor=2.0, cap=60.0) == 10.0
        assert bp.retry_delay_seconds(2, base=5.0, factor=2.0, cap=60.0) == 20.0

    def test_delay_is_capped(self):
        assert bp.retry_delay_seconds(10, base=5.0, factor=2.0, cap=60.0) == 60.0

    def test_retries_transient_then_succeeds(self):
        calls = []
        delays = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise _TransientError(code=91)
            return "ok"

        assert bp.run_with_mongo_retries(fn, attempts=4, sleep=delays.append) == "ok"
        assert len(calls) == 3
        assert delays == [bp.retry_delay_seconds(0), bp.retry_delay_seconds(1)]

    def test_non_transient_error_raises_without_retrying(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("permanent")

        with pytest.raises(ValueError):
            bp.run_with_mongo_retries(fn, attempts=4, sleep=lambda _: None)
        assert len(calls) == 1

    def test_exhausted_attempts_reraise(self):
        calls = []
        delays = []

        def fn():
            calls.append(1)
            raise _TransientError(code=116)

        with pytest.raises(_TransientError):
            bp.run_with_mongo_retries(fn, attempts=3, sleep=delays.append)
        assert len(calls) == 3
        assert len(delays) == 2


class TestWriteIsUpsertNotOverwrite:
    def test_write_options_express_upsert(self):
        opts = bp.feature_write_options("mongodb://example", "weather_features")
        assert opts["operationType"] == "replace"
        assert opts["upsertDocument"] == "true"
        assert opts["collection"] == "weather_features"
        assert "mode" not in opts

    def test_save_uses_append_mode_and_never_overwrite(self):
        # Inspect the actual `.mode(...)` calls, not prose: the docstring names
        # the old overwrite behaviour on purpose.
        tree = ast.parse(inspect.getsource(bp.save_features_to_mongodb))
        mode_args = [
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mode"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        assert "append" in mode_args
        assert "overwrite" not in mode_args

    def test_save_adds_the_stable_id_column(self):
        body = inspect.getsource(bp.save_features_to_mongodb)
        assert "add_feature_id" in body
