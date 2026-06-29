"""
conftest.py
===========
Shared fixtures for the weather-prediction-mlops test suite.

All Spark tests use master("local[1]") — no cluster needed in CI.
Mongo tests use mongomock to avoid a real connection.
"""

import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark():
    """Local SparkSession for unit tests (no cluster, no MongoDB connector)."""
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("weather-unit-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# MongoDB mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mongo_client():
    """Return a MagicMock that mimics MongoClient well enough for unit tests."""
    client = MagicMock()
    db = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)
    return client, db


# ---------------------------------------------------------------------------
# Sample document factories
# ---------------------------------------------------------------------------


def make_open_meteo_message(city: str = "Madrid", unix_ts: int = 1_750_000_000) -> dict:
    """Minimal Open-Meteo-shaped Kafka message value."""
    return {
        "city": city,
        "latitude": 40.4168,
        "longitude": -3.7038,
        "utc_offset_seconds": 3600,
        "timezone": "Europe/Madrid",
        "current": {
            "time": unix_ts,
            "temperature_2m": 22.5,
            "apparent_temperature": 21.0,
            "relative_humidity_2m": 55,
            "surface_pressure": 1013.0,
            "pressure_msl": 1015.0,
            "dew_point_2m": 13.5,
            "wind_speed_10m": 5.2,
            "wind_direction_10m": 270,
            "wind_gusts_10m": 8.1,
            "wind_speed_80m": 7.3,
            "wind_direction_80m": 265,
            "cloud_cover": 20,
            "precipitation": 0.0,
            "rain": 0.0,
            "showers": 0.0,
            "snowfall": 0.0,
            "visibility": 24140.0,
            "shortwave_radiation": 350.0,
            "cape": 0.0,
            "weather_code": 1,
            "uv_index": 4.0,
            "is_day": 1,
        },
    }
