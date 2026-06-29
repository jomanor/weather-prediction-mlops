"""
tests/unit/test_consumer.py
============================
Unit tests for kafka/kafka-consumer/consumer.py

We mock KafkaConsumer and MongoClient so no real infrastructure is needed.
"""

import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup — consumer.py lives outside the package root
# ---------------------------------------------------------------------------
CONSUMER_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "kafka", "kafka-consumer"
)
sys.path.insert(0, CONSUMER_DIR)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_consumer(mock_mongo_client):
    """Instantiate WeatherConsumer with all external deps mocked."""
    client, db = mock_mongo_client

    with (
        patch("consumer.KafkaConsumer"),
        patch("consumer.MongoClient", return_value=client),
        patch.dict(
            os.environ,
            {
                "KAFKA_BROKER": "localhost:9092",
                "MONGO_URL": "mongodb://localhost:27017",
            },
        ),
    ):
        from consumer import WeatherConsumer  # noqa: PLC0415

        c = WeatherConsumer()

    # Attach the mocks so tests can assert on them
    c.raw_collection = MagicMock()
    c.current_collection = MagicMock()
    return c


# ---------------------------------------------------------------------------
# _unix_to_datetime
# ---------------------------------------------------------------------------


class TestUnixToDatetime:
    def test_known_timestamp(self, mock_mongo_client):
        """1_750_000_000 → 2025-06-15 19:26:40 UTC"""
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.MongoClient"),
            patch.dict(os.environ, {"MONGO_URL": "x", "KAFKA_BROKER": "x"}),
        ):
            from consumer import WeatherConsumer  # noqa: PLC0415

            c = WeatherConsumer.__new__(WeatherConsumer)

        result = c._unix_to_datetime(1_750_000_000)
        assert result.tzinfo is not None, "Result must be timezone-aware"
        assert result == datetime(2025, 6, 15, 15, 6, 40, tzinfo=timezone.utc)

    def test_invalid_input_returns_now(self):
        with (
            patch("consumer.KafkaConsumer"),
            patch("consumer.MongoClient"),
            patch.dict(os.environ, {"MONGO_URL": "x", "KAFKA_BROKER": "x"}),
        ):
            from consumer import WeatherConsumer  # noqa: PLC0415

            c = WeatherConsumer.__new__(WeatherConsumer)

        result = c._unix_to_datetime("not-a-number")
        assert isinstance(result, datetime)


# ---------------------------------------------------------------------------
# process_message — idempotency
# ---------------------------------------------------------------------------


class TestProcessMessage:
    def _make_message(self, city="Madrid", unix_ts=1_750_000_000):
        """Mimics a kafka-python message object."""
        msg = MagicMock()
        msg.value = {
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
        return msg

    def test_upsert_called_for_raw_collection(self, mock_mongo_client):
        """process_message must call replace_one(upsert=True) on raw_weather."""
        c = _make_consumer(mock_mongo_client)
        msg = self._make_message()
        c.process_message(msg)
        c.raw_collection.replace_one.assert_called_once()
        _, kwargs = c.raw_collection.replace_one.call_args
        assert kwargs.get("upsert") is True

    def test_upsert_called_for_current_collection(self, mock_mongo_client):
        """process_message must call replace_one(upsert=True) on weather_data."""
        c = _make_consumer(mock_mongo_client)
        msg = self._make_message()
        c.process_message(msg)
        c.current_collection.replace_one.assert_called_once()
        _, kwargs = c.current_collection.replace_one.call_args
        assert kwargs.get("upsert") is True

    def test_duplicate_message_same_id(self, mock_mongo_client):
        """Sending the same message twice → replace_one called twice (upsert handles dedup in Mongo)."""
        c = _make_consumer(mock_mongo_client)
        msg = self._make_message(city="Barcelona", unix_ts=1_750_000_000)
        c.process_message(msg)
        c.process_message(msg)
        assert c.raw_collection.replace_one.call_count == 2

    def test_document_id_format(self, mock_mongo_client):
        """_id must follow the pattern '<city>_<unix_ts>'."""
        c = _make_consumer(mock_mongo_client)
        msg = self._make_message(city="Valencia", unix_ts=1_750_000_000)
        c.process_message(msg)

        raw_call_args = c.raw_collection.replace_one.call_args
        filter_doc = raw_call_args[0][0]
        expected_id = "Valencia_1750000000"
        assert filter_doc["_id"] == expected_id


# ---------------------------------------------------------------------------
# _weather_code_description
# ---------------------------------------------------------------------------


class TestWeatherCodeDescription:
    def test_known_code(self):
        from consumer import _weather_code_description  # noqa: PLC0415

        assert _weather_code_description(0) == "clear sky"
        assert _weather_code_description(95) == "thunderstorm"

    def test_unknown_code(self):
        from consumer import _weather_code_description  # noqa: PLC0415

        result = _weather_code_description(999)
        assert "999" in result
