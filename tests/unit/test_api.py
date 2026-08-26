"""
test_api.py
===========
Unit tests for FastAPI weather and predictions REST endpoints.
Uses AsyncMock / MagicMock on api.db to verify response formatting,
status codes, and open-meteo/legacy schema compatibility without
requiring a live MongoDB instance.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Make api module discoverable
API_DIR = os.path.join(os.path.dirname(__file__), "../../api")
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

import api as weather_api  # noqa: E402


@pytest.fixture()
def mock_db():
    """Mock MongoDB database attached to weather_api.db."""
    mock_database = MagicMock()
    mock_database.weather_data.create_index = AsyncMock()
    mock_database.weather_predictions.create_index = AsyncMock()
    return mock_database


@pytest.fixture()
def client(mock_db):
    """FastAPI TestClient with mocked lifespan / MongoDB client."""
    mock_motor = MagicMock()
    mock_motor.admin.command = AsyncMock(return_value={"ok": 1})
    mock_motor.__getitem__ = MagicMock(return_value=mock_db)

    with patch("api.AsyncIOMotorClient", return_value=mock_motor):
        original_db = weather_api.db
        original_client = weather_api.motor_client
        weather_api.db = mock_db
        weather_api.motor_client = mock_motor
        try:
            with TestClient(weather_api.app, raise_server_exceptions=False) as c:
                yield c
        finally:
            weather_api.db = original_db
            weather_api.motor_client = original_client


class TestRootEndpoint:
    def test_root_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["status"] == "healthy"
        assert "version" in json_data


class TestWeatherEndpoints:
    def test_get_current_weather_open_meteo_schema(self, client, mock_db):
        now = datetime.now(timezone.utc)
        mock_db.weather_data.find_one = AsyncMock(
            return_value={
                "city": "Madrid",
                "temperature": 23.5,
                "feels_like": 22.0,
                "humidity": 45,
                "pressure": 1014.0,
                "wind_speed": 4.1,
                "description": "Clear sky",
                "timestamp": now,
            }
        )

        response = client.get("/weather/current/Madrid")
        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "Madrid"
        assert data["temperature"] == 23.5
        assert data["description"] == "Clear sky"

    def test_get_current_weather_legacy_owm_schema(self, client, mock_db):
        now = datetime.now(timezone.utc)
        mock_db.weather_data.find_one = AsyncMock(
            return_value={
                "city": "Barcelona",
                "data": {
                    "main": {
                        "temp": 20.0,
                        "feels_like": 19.5,
                        "humidity": 60,
                        "pressure": 1012,
                    },
                    "wind": {"speed": 3.5},
                },
                "description": "Few clouds",
                "timestamp": now,
            }
        )

        response = client.get("/weather/current/Barcelona")
        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "Barcelona"
        assert data["temperature"] == 20.0
        assert data["humidity"] == 60

    def test_get_current_weather_not_found(self, client, mock_db):
        mock_db.weather_data.find_one = AsyncMock(return_value=None)
        response = client.get("/weather/current/Atlantis")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_all_cities(self, client, mock_db):
        mock_db.weather_data.distinct = AsyncMock(return_value=["Valencia", "Madrid", "Barcelona"])
        response = client.get("/weather/cities")
        assert response.status_code == 200
        assert response.json() == ["Barcelona", "Madrid", "Valencia"]

    def test_get_historical_weather(self, client, mock_db):
        now = datetime.now(timezone.utc)

        class AsyncCursorMock:
            def __init__(self, items):
                self.items = items

            def sort(self, *args, **kwargs):
                return self

            def limit(self, *args, **kwargs):
                return self

            def __aiter__(self):
                self._iter = iter(self.items)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        sample_docs = [
            {
                "city": "Sevilla",
                "temperature": 28.0,
                "feels_like": 27.5,
                "humidity": 35,
                "pressure": 1010.0,
                "wind_speed": 2.0,
                "description": "Sunny",
                "timestamp": now,
            }
        ]
        mock_db.weather_data.find = MagicMock(return_value=AsyncCursorMock(sample_docs))

        response = client.get("/weather/historical/Sevilla?hours=12")
        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "Sevilla"
        assert data["count"] == 1
        assert data["data"][0]["temperature"] == 28.0

    def test_compare_cities_bad_request(self, client, mock_db):
        response = client.get("/weather/compare?cities=Madrid")
        assert response.status_code == 400
        assert "at least 2 cities" in response.json()["detail"]


class TestPredictionsEndpoints:
    def test_get_latest_predictions(self, client, mock_db):
        now = datetime.now(timezone.utc)

        class AsyncAggCursorMock:
            def __init__(self, items):
                self.items = items

            def __aiter__(self):
                self._iter = iter(self.items)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        predictions_sample = [
            {
                "city": "Madrid",
                "source_timestamp": now,
                "prediction_timestamp": now,
                "predicted_temperature": 24.2,
                "predicted_rain": 0.0,
                "observed_temperature": 23.5,
                "horizon_hours": 1,
                "temp_model_name": "temp_prediction_1h",
                "temp_model_version": "v1",
                "rain_model_name": "rain_prediction_1h",
                "rain_model_version": "v1",
            }
        ]
        mock_db.weather_predictions.aggregate = MagicMock(
            return_value=AsyncAggCursorMock(predictions_sample)
        )

        response = client.get("/predictions/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["predictions"][0]["city"] == "Madrid"
        assert data["predictions"][0]["predicted_temperature"] == 24.2

    def test_get_predictions_for_city_not_found(self, client, mock_db):
        class EmptyAsyncCursorMock:
            def sort(self, *args, **kwargs):
                return self

            def limit(self, *args, **kwargs):
                return self

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        mock_db.weather_predictions.find = MagicMock(return_value=EmptyAsyncCursorMock())

        response = client.get("/predictions/NonExistentCity")
        assert response.status_code == 404
        assert "No predictions found" in response.json()["detail"]
