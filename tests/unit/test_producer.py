"""
tests/unit/test_producer.py
============================
Unit tests for kafka/kafka-producer/producer.py

We mock requests.get and KafkaProducer so no network calls happen.
"""

import sys
import os
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PRODUCER_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "kafka", "kafka-producer"
)
sys.path.insert(0, PRODUCER_DIR)


def _make_producer():
    with (
        patch("producer.KafkaProducer"),
        patch.dict(os.environ, {"KAFKA_BROKER": "localhost:9092"}),
    ):
        from producer import WeatherProducer  # noqa: PLC0415
        return WeatherProducer()


# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------

class TestCities:
    def test_has_14_cities(self):
        p = _make_producer()
        assert len(p.cities) == 14

    def test_madrid_present(self):
        p = _make_producer()
        assert "Madrid" in p.cities

    def test_all_coords_are_tuples_of_floats(self):
        p = _make_producer()
        for city, (lat, lon) in p.cities.items():
            assert isinstance(lat, float), f"{city}: lat must be float"
            assert isinstance(lon, float), f"{city}: lon must be float"

    def test_iberian_latitudes_in_range(self):
        """All cities are on/around the Iberian peninsula: 35°N–44°N, 9°W–4°E."""
        p = _make_producer()
        for city, (lat, lon) in p.cities.items():
            assert 35 <= lat <= 44, f"{city} lat={lat} out of expected range"
            assert -9 <= lon <= 4, f"{city} lon={lon} out of expected range"


# ---------------------------------------------------------------------------
# Pressure levels
# ---------------------------------------------------------------------------

class TestPressureLevels:
    def test_has_6_pressure_levels(self):
        p = _make_producer()
        assert len(p.pressure_levels) == 6

    def test_expected_levels(self):
        p = _make_producer()
        assert set(p.pressure_levels) == {200, 500, 700, 850, 925, 1000}


# ---------------------------------------------------------------------------
# fetch_weather — request params
# ---------------------------------------------------------------------------

class TestFetchWeather:
    def _mock_response(self, city="Madrid"):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"city": city, "current": {}, "hourly": {}}
        return resp

    def test_includes_hourly_pressure_level(self):
        """The request must include hourly_pressure_level in params."""
        p = _make_producer()
        with patch("producer.requests.get", return_value=self._mock_response()) as mock_get:
            p.fetch_weather("Madrid", (40.4168, -3.7038))

        _, kwargs = mock_get.call_args
        params = kwargs.get("params", mock_get.call_args[0][1] if len(mock_get.call_args[0]) > 1 else {})
        assert "hourly_pressure_level" in params, "hourly_pressure_level must be in API params"

    def test_pressure_level_param_has_6_levels(self):
        p = _make_producer()
        with patch("producer.requests.get", return_value=self._mock_response()) as mock_get:
            p.fetch_weather("Madrid", (40.4168, -3.7038))

        _, kwargs = mock_get.call_args
        params = kwargs.get("params", {})
        assert len(params.get("pressure_level", [])) == 6

    def test_returns_none_on_http_error(self):
        p = _make_producer()
        bad_resp = MagicMock()
        bad_resp.status_code = 429
        bad_resp.text = "Too many requests"
        with patch("producer.requests.get", return_value=bad_resp):
            result = p.fetch_weather("Madrid", (40.4168, -3.7038))
        assert result is None

    def test_returns_none_on_exception(self):
        p = _make_producer()
        with patch("producer.requests.get", side_effect=ConnectionError("timeout")):
            result = p.fetch_weather("Madrid", (40.4168, -3.7038))
        assert result is None

    def test_city_injected_into_response(self):
        p = _make_producer()
        with patch("producer.requests.get", return_value=self._mock_response("Sevilla")):
            result = p.fetch_weather("Sevilla", (37.3891, -5.9845))
        assert result is not None
        assert result["city"] == "Sevilla"

    def test_uses_unixtime_format(self):
        """timeformat must be 'unixtime' so consumer can parse timestamps correctly."""
        p = _make_producer()
        with patch("producer.requests.get", return_value=self._mock_response()) as mock_get:
            p.fetch_weather("Madrid", (40.4168, -3.7038))

        _, kwargs = mock_get.call_args
        params = kwargs.get("params", {})
        assert params.get("timeformat") == "unixtime"
