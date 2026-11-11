"""Repository mapping: stored wind is m/s, the contract is km/h."""

from datetime import datetime, timedelta, timezone

from app.core.cities import DEFAULT_CITIES
from app.core.thresholds import QUALITY_OK_AGE_HOURS, QUALITY_WARN_AGE_HOURS
from app.repositories.weather_repo import (
    _CURRENT_FIELD_PATHS,
    _CURRENT_IDENTITY,
    _RAW_FIELD_PATHS,
    _RAW_IDENTITY,
    MS_TO_KMH,
    WeatherRepository,
    _downsample_by_step,
    _from_raw_weather,
    _from_weather_data,
    _projection,
    _quality_status,
)
from app.schemas.weather import WEATHER_FIELDS, CurrentWeather
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


def test_the_contract_value_is_18_kmh_for_5_ms():
    """Pinned to the number, not to ``MS_TO_KMH``.

    A test written as ``5.0 * MS_TO_KMH`` reads the constant it is supposed to
    guard, so halving the constant leaves it green. The contract says 5 m/s is
    18 km/h on the wire, which is the only thing a client can observe.
    """
    raw = _from_raw_weather(
        {
            "city": "Madrid",
            "timestamp": OBSERVED_AT,
            "payload": {"current": {"wind_speed_10m": 5.0}},
        }
    )
    legacy = _from_weather_data(
        {"city": "Madrid", "timestamp": OBSERVED_AT, "data": {"wind": {"speed": 5.0}}}
    )

    assert raw is not None and legacy is not None
    assert raw.wind_speed == 18.0
    assert legacy.wind_speed == 18.0


def test_the_conversion_has_one_authority():
    """Both repositories read the same 3.6, from ``app.core.units``."""
    from app.core import units
    from app.repositories import analytics_repo as analytics
    from app.repositories import weather_repo as weather

    assert units.MS_TO_KMH == 3.6
    assert weather.MS_TO_KMH is units.MS_TO_KMH
    assert analytics.MS_TO_KMH is units.MS_TO_KMH


# ---------------------------------------------------------------------------
# ``is_day``: only raw_weather carries it (``payload.current.is_day``)
# ---------------------------------------------------------------------------


def test_raw_weather_is_day_round_trips_for_both_values():
    for value in (0, 1):
        doc = {
            "city": "Madrid",
            "timestamp": OBSERVED_AT,
            "payload": {"current": {"is_day": value}},
        }

        point = _from_raw_weather(doc)

        assert point is not None
        assert point.is_day == value
        # The field is part of the public payload, so it must survive serialization.
        assert point.model_dump(mode="json")["is_day"] == value


def test_missing_is_day_stays_none():
    raw = _from_raw_weather({"city": "Madrid", "timestamp": OBSERVED_AT})
    legacy = _from_weather_data({"city": "Madrid", "timestamp": OBSERVED_AT})

    assert raw is not None and raw.is_day is None
    # ``weather_data`` never stored it: legacy rows serialize ``is_day: null``.
    assert legacy is not None and legacy.is_day is None
    assert legacy.model_dump(mode="json")["is_day"] is None


def test_is_day_is_a_selectable_field_and_kept_by_the_raw_projection():
    assert "is_day" in WEATHER_FIELDS
    projection = _projection(["is_day"], _RAW_FIELD_PATHS, _RAW_IDENTITY)
    assert projection is not None
    assert projection["payload.current.is_day"] == 1
    # No legacy spelling exists, so requesting it from ``weather_data`` adds no path.
    assert _projection(["is_day"], _CURRENT_FIELD_PATHS, _CURRENT_IDENTITY) == {
        "city": 1,
        "timestamp": 1,
        "latitude": 1,
        "longitude": 1,
    }


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
    raw = FakeMongoCollection(
        [
            {
                "city": "Madrid",
                "timestamp": OBSERVED_AT,
                "payload": {
                    "latitude": 40.4168,
                    "longitude": -3.7038,
                    "current": {"temperature_2m": 21.0},
                },
            }
        ]
    )
    current = FakeMongoCollection([])
    repo = WeatherRepository(FakeMongoDb(current, raw))

    points = await repo.find_many_in_range(
        ["Madrid"],
        OBSERVED_AT - timedelta(hours=1),
        OBSERVED_AT + timedelta(hours=1),
        fields=["temperature"],
    )

    assert [point.temperature for point in points] == [21.0]
    assert points[0].latitude == 40.4168 and points[0].longitude == -3.7038
    query, projection = raw.find_calls[0]
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
        "payload.latitude": 1,
        "payload.longitude": 1,
        "payload.current.temperature_2m": 1,
    }
    # Must match ``{city: 1, timestamp: -1}``: this is what keeps the query
    # index-fed instead of forcing a blocking in-memory SORT.
    assert raw.cursors[0].sort_spec == [("city", 1), ("timestamp", -1)]
    assert current.find_calls == []  # no fallback when raw_weather has the city


async def test_find_many_in_range_falls_back_to_legacy_per_missing_city():
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
    current = FakeMongoCollection(
        [{"city": "Madrid", "timestamp": OBSERVED_AT, "data": {"main": {"temp": 21.0}}}]
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
    assert raw_query["city"] == {"$in": ["Madrid", "Alicante"]}
    # Only the city raw_weather could not serve is re-queried in the legacy
    # collection, and that follow-up keeps its city-prefixed sort too.
    current_query, _ = current.find_calls[0]
    assert current_query["city"] == {"$in": ["Madrid"]}
    assert current.cursors[0].sort_spec == [("city", 1), ("timestamp", -1)]


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


async def test_raw_only_data_keeps_its_coordinates_under_projection():
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


# ---------------------------------------------------------------------------
# Data-quality meter (Contract 1): weather_features, city-prefixed sort
# ---------------------------------------------------------------------------


def _feature(city: str, hour: float, temperature: float | None = 20.0) -> dict:
    return {
        "city": city,
        "timestamp": OBSERVED_AT - timedelta(hours=hour),
        "temperature": temperature,
    }


async def test_quality_report_per_city_queries_are_city_prefixed_and_sorted():
    features = FakeMongoCollection(
        [
            _feature("Madrid", 0, 20.0),
            _feature("Madrid", 1, None),
            _feature("Madrid", 3, 22.0),
            _feature("Madrid", 4, 23.0),
            _feature("Alicante", 0, 30.0),
            _feature("Alicante", 1, 30.5),
        ]
    )
    repo = WeatherRepository(FakeMongoDb(weather_features=features))

    report = await repo.quality_report(days=1, now=OBSERVED_AT)

    # Seeded from the fixed 14-station registry, never ``distinct``: a city
    # with zero feature rows must still appear (see the outage test below).
    expected_cities = sorted(name for name, _, _ in DEFAULT_CITIES)
    assert [city.city for city in report] == expected_cities
    assert len(expected_cities) == 14
    # One equality-prefixed query per city: ``city`` is pinned, so the
    # ``{city: 1, timestamp: -1}`` index serves the ascending timestamp sort.
    # Every city without a row in the window also gets a descending, still
    # city-prefixed, latest-row fallback (12 of the 14 stations here).
    window_cursors = [
        cursor for cursor in features.cursors if cursor.sort_spec == [("city", 1), ("timestamp", 1)]
    ]
    fallback_cursors = [
        cursor
        for cursor in features.cursors
        if cursor.sort_spec == [("city", 1), ("timestamp", -1)]
    ]
    assert len(window_cursors) == len(expected_cities)
    assert len(fallback_cursors) == len(expected_cities) - 2
    assert all(cursor.sort_spec[0] == ("city", 1) for cursor in features.cursors)
    assert features.find_calls[0][0] == {
        "city": "Alicante",
        "timestamp": {"$gte": OBSERVED_AT - timedelta(days=1), "$lte": OBSERVED_AT},
    }
    assert features.find_calls[0][1] == {"city": 1, "timestamp": 1, "temperature": 1}


async def test_quality_report_computes_gaps_nulls_completeness_and_age():
    features = FakeMongoCollection(
        [
            _feature("Madrid", hour, temperature)
            for hour, temperature in ((0, 13.0), (1, 12.0), (3, None), (4, 10.0))
        ]
    )
    repo = WeatherRepository(FakeMongoDb(weather_features=features))

    report = {city.city: city for city in await repo.quality_report(days=1, now=OBSERVED_AT)}
    city = report["Madrid"]

    assert city.city == "Madrid"
    assert city.expected_hours == 24
    assert city.observed_hours == 4
    assert city.completeness == 4 / 24
    # Observed at -4h, -3h, -1h, now -> gaps of 1 h, 2 h, 1 h.
    assert city.max_gap_hours == 2.0
    assert city.null_rate == 0.25
    assert city.last_observed_at == OBSERVED_AT
    assert city.age_hours == 0.0
    assert city.status == "bad"  # completeness 0.17 dominates a fresh timestamp


async def test_quality_report_dedupes_duplicate_timestamps():
    features = FakeMongoCollection(
        [
            _feature("Madrid", 0, 20.0),
            _feature("Madrid", 0, 22.0),  # same hour written twice
            _feature("Madrid", 2, 21.0),
        ]
    )
    repo = WeatherRepository(FakeMongoDb(weather_features=features))

    report = {city.city: city for city in await repo.quality_report(days=1, now=OBSERVED_AT)}
    madrid = report["Madrid"]

    assert madrid.observed_hours == 2  # two distinct hours, not three rows
    assert madrid.max_gap_hours == 2.0  # not collapsed to 0 by the duplicate


async def test_quality_report_city_with_zero_features_is_bad():
    # ``weather_features`` only ever has Madrid; every other station in the
    # canonical registry has vanished entirely and must be reported, not dropped.
    features = FakeMongoCollection([_feature("Madrid", 0, 20.0)])
    repo = WeatherRepository(FakeMongoDb(weather_features=features))

    report = {city.city: city for city in await repo.quality_report(days=1, now=OBSERVED_AT)}

    assert len(report) == 14
    missing = report["Barcelona"]
    assert missing.observed_hours == 0
    assert missing.completeness == 0.0
    assert missing.null_rate is None
    assert missing.last_observed_at is None
    assert missing.age_hours is None
    assert missing.status == "bad"


async def test_quality_report_city_outside_window_is_empty_and_bad():
    features = FakeMongoCollection(
        [
            _feature("Madrid", 100, 10.0),  # far outside a 1-day window
            _feature("Alicante", 1, 30.0),
        ]
    )
    repo = WeatherRepository(FakeMongoDb(weather_features=features))

    report = {city.city: city for city in await repo.quality_report(days=1, now=OBSERVED_AT)}

    madrid = report["Madrid"]
    assert madrid.observed_hours == 0
    assert madrid.completeness == 0.0
    assert madrid.max_gap_hours == 0.0  # fewer than two points
    assert madrid.null_rate is None
    # The window is empty but the station is not: report its real age instead
    # of null (which would claim it was never observed).
    assert madrid.last_observed_at == OBSERVED_AT - timedelta(hours=100)
    assert madrid.age_hours == 100.0
    assert madrid.status == "bad"


async def test_quality_report_completeness_is_clamped_to_one():
    features = FakeMongoCollection(
        [_feature("Madrid", hour) for hour in range(25)]  # 25 rows in a 1-day window
    )
    repo = WeatherRepository(FakeMongoDb(weather_features=features))

    report = {city.city: city for city in await repo.quality_report(days=1, now=OBSERVED_AT)}

    assert report["Madrid"].observed_hours == 25
    assert report["Madrid"].completeness == 1.0


def test_quality_status_worst_metric_wins():
    assert _quality_status(0.99, 1.0) == "ok"
    assert _quality_status(0.90, 1.0) == "warn"  # completeness-only downgrade
    assert _quality_status(0.99, 13.0) == "warn"  # age-only downgrade
    assert _quality_status(0.50, 1.0) == "bad"
    assert _quality_status(0.99, None) == "bad"  # never observed


def test_quality_age_thresholds_match_the_rebuild_cycle():
    # Batch 4, Contract 5: ~2x / ~3x the ~6 h feature-rebuild cycle. The values
    # are the shared ``app.core.thresholds`` ones (audit finding 5): the meter
    # must not carry a second copy that can drift from the watchdog's.
    from app.core import thresholds
    from app.repositories import weather_repo

    assert (QUALITY_OK_AGE_HOURS, QUALITY_WARN_AGE_HOURS) == (12.0, 18.0)
    assert weather_repo.QUALITY_OK_AGE_HOURS is thresholds.QUALITY_OK_AGE_HOURS
    assert weather_repo.QUALITY_WARN_AGE_HOURS is thresholds.QUALITY_WARN_AGE_HOURS


def test_quality_status_grades_at_exact_thresholds():
    assert _quality_status(0.95, 12.0) == "ok"
    assert _quality_status(0.80, 18.0) == "warn"
    assert _quality_status(0.7999, 12.0) == "bad"
    assert _quality_status(0.95, 12.0001) == "warn"
    assert _quality_status(0.95, None) == "bad"  # missing last observation


# ---------------------------------------------------------------------------
# Read order (audit finding 3): ``raw_weather`` first, ``weather_data`` only as
# the fallback. Reading the legacy collection first meant ONE leftover document
# inside its 30-day TTL could replace fourteen live stations and shrink a
# history to a single point.
# ---------------------------------------------------------------------------


class _RecordingCollection(FakeMongoCollection):
    """A collection double that appends every read to a shared call log."""

    def __init__(self, label: str, log: list[str], documents: list[dict] | None = None) -> None:
        super().__init__(documents)
        self._label = label
        self._log = log

    def find(self, query, projection=None):
        self._log.append(f"{self._label}.find")
        return super().find(query, projection)

    def aggregate(self, pipeline, allowDiskUse=False, **kwargs):
        self._log.append(f"{self._label}.aggregate")
        return super().aggregate(pipeline, allowDiskUse=allowDiskUse, **kwargs)

    async def find_one(self, query, sort=None):
        self._log.append(f"{self._label}.find_one")
        return await super().find_one(query, sort=sort)


def _raw_doc(city: str, hour: float, temperature: float) -> dict:
    return {
        "city": city,
        "timestamp": OBSERVED_AT - timedelta(hours=hour),
        "payload": {
            "latitude": 40.0,
            "longitude": -3.0,
            "current": {"temperature_2m": temperature},
        },
    }


def _legacy_doc(city: str, hour: float, temperature: float) -> dict:
    return {
        "city": city,
        "timestamp": OBSERVED_AT - timedelta(hours=hour),
        "data": {"main": {"temp": temperature}},
    }


def _repo(log: list[str], raw_docs: list[dict], legacy_docs: list[dict]) -> WeatherRepository:
    """A repository over a ``raw_weather`` double and a ``weather_data`` double."""
    return WeatherRepository(
        FakeMongoDb(
            _RecordingCollection("legacy", log, legacy_docs),
            _RecordingCollection("raw", log, raw_docs),
        )
    )


async def test_latest_for_city_reads_raw_first_and_never_touches_legacy():
    log: list[str] = []
    repo = _repo(log, [_raw_doc("Madrid", 0, 22.0)], [_legacy_doc("Madrid", 5, 10.0)])

    point = await repo.latest_for_city("Madrid")

    assert point is not None and point.temperature == 22.0  # raw wins, not the stale doc
    assert log == ["raw.find_one"]  # one round-trip: the fallback never ran


async def test_latest_for_city_falls_back_to_legacy_when_raw_is_empty():
    log: list[str] = []
    repo = _repo(log, [], [_legacy_doc("Madrid", 5, 10.0)])

    point = await repo.latest_for_city("Madrid")

    # The fallback exists precisely so legacy-only data is still served.
    assert point is not None and point.temperature == 10.0
    assert log == ["raw.find_one", "legacy.find_one"]


async def test_latest_per_city_aggregates_raw_first_with_disk_use_and_city_sort():
    log: list[str] = []
    raw = _RecordingCollection(
        "raw", log, [_raw_doc("Madrid", 0, 22.0), _raw_doc("Alicante", 0, 30.0)]
    )
    current = _RecordingCollection("legacy", log, [_legacy_doc("Sevilla", 0, 5.0)])
    repo = WeatherRepository(FakeMongoDb(current, raw))

    stations = await repo.latest_per_city()

    # One stale legacy document must not replace the live stations.
    assert [(station.city, station.temperature) for station in stations] == [
        ("Alicante", 30.0),
        ("Madrid", 22.0),
    ]
    assert log == ["raw.aggregate"]
    aggregate = raw.aggregate_calls[0]
    assert aggregate["allowDiskUse"] is True
    assert aggregate["pipeline"][0] == {"$sort": {"city": 1, "timestamp": -1}}


async def test_latest_per_city_falls_back_to_legacy_only_when_raw_has_no_rows():
    log: list[str] = []
    raw = _RecordingCollection("raw", log, [])
    current = _RecordingCollection("legacy", log, [_legacy_doc("Sevilla", 0, 5.0)])
    repo = WeatherRepository(FakeMongoDb(current, raw))

    stations = await repo.latest_per_city()

    assert [(station.city, station.temperature) for station in stations] == [("Sevilla", 5.0)]
    assert log == ["raw.aggregate", "legacy.aggregate"]
    assert current.aggregate_calls[0]["allowDiskUse"] is True


async def test_find_range_reads_raw_first_and_keeps_every_point():
    log: list[str] = []
    repo = _repo(
        log,
        [_raw_doc("Madrid", hour, 20.0 + hour) for hour in range(3)],
        [_legacy_doc("Madrid", 1, 99.0)],
    )

    points = await repo.find_range(
        "Madrid", OBSERVED_AT - timedelta(hours=2), OBSERVED_AT, newest_first=False
    )

    # Three raw points, not the single legacy one.
    assert [point.temperature for point in points] == [22.0, 21.0, 20.0]
    assert log == ["raw.find"]


async def test_find_range_falls_back_to_legacy_when_raw_window_is_empty():
    log: list[str] = []
    repo = _repo(log, [_raw_doc("Madrid", 100, 1.0)], [_legacy_doc("Madrid", 1, 10.0)])

    points = await repo.find_range(
        "Madrid", OBSERVED_AT - timedelta(hours=2), OBSERVED_AT, newest_first=False
    )

    assert [point.temperature for point in points] == [10.0]
    assert log == ["raw.find", "legacy.find"]


async def test_find_many_in_range_reads_raw_before_legacy():
    log: list[str] = []
    repo = _repo(log, [_raw_doc("Madrid", 0, 22.0)], [_legacy_doc("Alicante", 0, 30.0)])

    points = await repo.find_many_in_range(
        ["Madrid", "Alicante"], OBSERVED_AT - timedelta(hours=1), OBSERVED_AT
    )

    assert [(point.city, point.temperature) for point in points] == [
        ("Alicante", 30.0),
        ("Madrid", 22.0),
    ]
    # raw first, legacy only for the city raw could not serve.
    assert log == ["raw.find", "legacy.find"]
