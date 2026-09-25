"""Repository mapping: stored wind is m/s, the contract is km/h."""

from datetime import datetime, timezone

from app.repositories.weather_repo import (
    MS_TO_KMH,
    _from_raw_weather,
    _from_weather_data,
)

UTC = timezone.utc
OBSERVED_AT = datetime(2026, 9, 23, 12, tzinfo=UTC)


def test_weather_data_wind_is_converted_from_ms_to_kmh():
    doc = {
        "city": "Madrid",
        "timestamp": OBSERVED_AT,
        "data": {"wind": {"speed": 5.0, "deg": 200.0}, "weather": [{"id": 2}]},
    }

    point = _from_weather_data(doc)

    assert point is not None
    assert point.wind_speed == 5.0 * MS_TO_KMH
    assert point.wind_direction == 200.0


def test_raw_weather_wind_is_converted_from_ms_to_kmh():
    doc = {
        "city": "Madrid",
        "timestamp": OBSERVED_AT,
        "payload": {"current": {"wind_speed_10m": 10.0, "wind_direction_10m": 200.0}},
    }

    point = _from_raw_weather(doc)

    assert point is not None
    assert point.wind_speed == 10.0 * MS_TO_KMH
    assert point.wind_direction == 200.0


def test_missing_wind_stays_none():
    assert _from_weather_data({"city": "Madrid", "timestamp": OBSERVED_AT}).wind_speed is None
    assert _from_raw_weather({"city": "Madrid", "timestamp": OBSERVED_AT}).wind_speed is None
