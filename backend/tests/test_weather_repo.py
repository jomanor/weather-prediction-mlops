"""Repository mapping: stored wind is m/s, the contract is km/h."""

from datetime import datetime, timedelta, timezone

from app.repositories.weather_repo import (
    _CURRENT_IDENTITY,
    _RAW_IDENTITY,
    MS_TO_KMH,
    WeatherRepository,
    _downsample_by_step,
    _from_raw_weather,
    _from_weather_data,
    _projection,
)
from app.schemas.weather import CurrentWeather
from tests.fakes import FakeMongoCollection, FakeMongoDb

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


# ---------------------------------------------------------------------------
# Bulk read: projection, city-prefixed sort, raw fallback, step downsampling
# ---------------------------------------------------------------------------


def test_projection_limits_to_requested_fields_plus_collection_identity():
    projection = _projection(
        ["temperature"],
        {"temperature": ("temperature", "data.main.temp")},
        _CURRENT_IDENTITY,
    )
    assert projection == {
        "city": 1,
        "timestamp": 1,
        "latitude": 1,
        "longitude": 1,
        "temperature": 1,
        "data.main.temp": 1,
    }
    # raw_weather nests the coordinates under payload.*, so the identity differs.
    raw_projection = _projection(["temperature"], {}, _RAW_IDENTITY)
    assert raw_projection == {
        "city": 1,
        "timestamp": 1,
        "payload.latitude": 1,
        "payload.longitude": 1,
    }
    assert _projection(None, {}, _CURRENT_IDENTITY) is None


async def test_find_many_in_range_projects_and_sorts_descending_timestamp():
    current = FakeMongoCollection(
        [
            {
                "city": "Madrid",
                "timestamp": OBSERVED_AT,
                "latitude": 40.4168,
                "longitude": -3.7038,
                "data": {"main": {"temp": 21.0}},
            }
        ]
    )
    raw = FakeMongoCollection([])
    repo = WeatherRepository(FakeMongoDb(current, raw))

    points = await repo.find_many_in_range(
        ["Madrid"],
        OBSERVED_AT - timedelta(hours=1),
        OBSERVED_AT + timedelta(hours=1),
        fields=["temperature"],
    )

    assert [point.temperature for point in points] == [21.0]
    assert points[0].latitude == 40.4168 and points[0].longitude == -3.7038
    query, projection = current.find_calls[0]
    assert query == {
        "city": {"$in": ["Madrid"]},
        "timestamp": {
            "$gte": OBSERVED_AT - timedelta(hours=1),
            "$lte": OBSERVED_AT + timedelta(hours=1),
        },
    }
    assert projection == {
        "city": 1,
        "timestamp": 1,
        "latitude": 1,
        "longitude": 1,
        "temperature": 1,
        "data.main.temp": 1,
    }
    # Must match ``{city: 1, timestamp: -1}``: this is what keeps the query
    # index-fed instead of forcing a blocking in-memory SORT.
    assert current.cursors[0].sort_spec == [("city", 1), ("timestamp", -1)]
    assert raw.find_calls == []  # no fallback when weather_data has the city


async def test_find_many_in_range_falls_back_to_raw_per_missing_city():
    current = FakeMongoCollection(
        [{"city": "Madrid", "timestamp": OBSERVED_AT, "data": {"main": {"temp": 21.0}}}]
    )
    raw = FakeMongoCollection(
        [
            {
                "city": "Alicante",
                "timestamp": OBSERVED_AT,
                "payload": {
                    "latitude": 38.3452,
                    "longitude": -0.481,
                    "current": {"temperature_2m": 25.0},
                },
            }
        ]
    )
    repo = WeatherRepository(FakeMongoDb(current, raw))

    points = await repo.find_many_in_range(
        ["Madrid", "Alicante"],
        OBSERVED_AT - timedelta(hours=1),
        OBSERVED_AT + timedelta(hours=1),
    )

    assert [(point.city, point.temperature) for point in points] == [
        ("Alicante", 25.0),
        ("Madrid", 21.0),
    ]
    raw_query, _ = raw.find_calls[0]
    assert raw_query["city"] == {"$in": ["Alicante"]}
    assert raw.cursors[0].sort_spec == [("city", 1), ("timestamp", -1)]


async def test_find_many_in_range_applies_in_and_range_filters():
    in_window = {
        "city": "Madrid",
        "timestamp": OBSERVED_AT,
        "data": {"main": {"temp": 21.0}},
    }
    before_window = {
        "city": "Madrid",
        "timestamp": OBSERVED_AT - timedelta(hours=5),
        "data": {"main": {"temp": 10.0}},
    }
    other_city = {
        "city": "Alicante",
        "timestamp": OBSERVED_AT,
        "data": {"main": {"temp": 30.0}},
    }
    current = FakeMongoCollection([before_window, other_city, in_window])
    repo = WeatherRepository(FakeMongoDb(current, FakeMongoCollection()))

    points = await repo.find_many_in_range(
        ["Madrid", "Alicante"],
        OBSERVED_AT - timedelta(hours=1),
        OBSERVED_AT + timedelta(hours=1),
    )

    # ``$in`` selects both cities, the timestamp window drops only the old row.
    assert [(point.city, point.temperature) for point in points] == [
        ("Alicante", 30.0),
        ("Madrid", 21.0),
    ]


async def test_raw_fallback_keeps_coordinates_under_projection():
    raw = FakeMongoCollection(
        [
            {
                "city": "Alicante",
                "timestamp": OBSERVED_AT,
                "payload": {
                    "latitude": 38.3452,
                    "longitude": -0.481,
                    "current": {"temperature_2m": 25.0},
                },
            }
        ]
    )
    repo = WeatherRepository(FakeMongoDb(FakeMongoCollection(), raw))

    points = await repo.find_many_in_range(
        ["Alicante"],
        OBSERVED_AT - timedelta(hours=1),
        OBSERVED_AT + timedelta(hours=1),
        fields=["temperature"],
    )

    assert points[0].latitude == 38.3452
    assert points[0].longitude == -0.481
    _, projection = raw.find_calls[0]
    assert projection["payload.latitude"] == 1
    assert projection["payload.longitude"] == 1


def _point(city: str, hour: int) -> CurrentWeather:
    return CurrentWeather(city=city, observed_at=OBSERVED_AT - timedelta(hours=hour))


def test_downsample_keeps_first_point_per_bucket_and_noop_for_step_one():
    # Ascending in time: 08:00 .. 12:00. Even UTC hours are 2 h bucket boundaries.
    points = [_point("Madrid", hour) for hour in range(4, -1, -1)]

    assert _downsample_by_step(points, 1) == points

    sampled = _downsample_by_step(points, 2)
    assert [point.observed_at for point in sampled] == [
        OBSERVED_AT - timedelta(hours=4),
        OBSERVED_AT - timedelta(hours=2),
        OBSERVED_AT,
    ]
