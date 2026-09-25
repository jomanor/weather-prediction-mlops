"""Repository mapping: stored wind is m/s, the contract is km/h."""

from datetime import datetime, timedelta, timezone

from app.repositories.weather_repo import (
    MS_TO_KMH,
    WeatherRepository,
    _downsample_by_step,
    _from_raw_weather,
    _from_weather_data,
    _projection,
)
from app.schemas.weather import CurrentWeather

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


class _Cursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    def sort(self, _spec):
        return self

    def limit(self, _count):
        return self

    def __aiter__(self):
        self._iterator = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None


class _Collection:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.find_calls: list[tuple[dict, dict | None]] = []

    def find(self, query, projection=None):
        self.find_calls.append((query, projection))
        return _Cursor(self.docs)


class _Db:
    def __init__(self, weather_data: _Collection, raw_weather: _Collection) -> None:
        self._collections = {"weather_data": weather_data, "raw_weather": raw_weather}

    def __getitem__(self, name):
        return self._collections[name]


def test_projection_limits_to_requested_fields_plus_identity():
    projection = _projection(
        ["temperature"],
        {
            "temperature": ("temperature", "data.main.temp"),
        },
    )
    assert projection == {"city": 1, "timestamp": 1, "temperature": 1, "data.main.temp": 1}
    assert _projection(None, {}) is None


async def test_find_many_in_range_projects_and_sorts_city_first():
    current = _Collection(
        [
            {
                "city": "Madrid",
                "timestamp": OBSERVED_AT,
                "data": {"main": {"temp": 21.0}},
            }
        ]
    )
    raw = _Collection([])
    repo = WeatherRepository(_Db(current, raw))

    points = await repo.find_many_in_range(
        ["Madrid"],
        OBSERVED_AT - timedelta(hours=1),
        OBSERVED_AT + timedelta(hours=1),
        fields=["temperature"],
    )

    assert [point.temperature for point in points] == [21.0]
    query, projection = current.find_calls[0]
    assert query == {
        "city": {"$in": ["Madrid"]},
        "timestamp": {
            "$gte": OBSERVED_AT - timedelta(hours=1),
            "$lte": OBSERVED_AT + timedelta(hours=1),
        },
    }
    assert projection == {"city": 1, "timestamp": 1, "temperature": 1, "data.main.temp": 1}
    assert raw.find_calls == []  # no fallback when weather_data has the city


async def test_find_many_in_range_falls_back_to_raw_per_missing_city():
    current = _Collection(
        [{"city": "Madrid", "timestamp": OBSERVED_AT, "data": {"main": {"temp": 21.0}}}]
    )
    raw = _Collection(
        [
            {
                "city": "Alicante",
                "timestamp": OBSERVED_AT,
                "payload": {"current": {"temperature_2m": 25.0}},
            }
        ]
    )
    repo = WeatherRepository(_Db(current, raw))

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
