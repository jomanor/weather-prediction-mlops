"""The ``cities`` registry is the runtime station set (audit finding 4).

``DEFAULT_CITIES`` is seed data only. Before the fix the quality meter iterated
that code tuple, so a station deleted through ``DELETE /cities`` reported
``bad`` forever while a newly registered one was never reported at all. These
tests pin the registry (a fake ``cities`` collection with two stations) as the
only runtime source.
"""

from datetime import datetime, timezone

from app.repositories.weather_repo import WeatherRepository
from tests.fakes import FakeMongoCollection, FakeMongoDb

OBSERVED_AT = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def _registry(*names: str) -> FakeMongoCollection:
    """A ``cities`` collection holding just these stations."""
    return FakeMongoCollection([{"name": name, "name_key": name.casefold()} for name in names])


def _feature(city: str, temperature: float = 20.0) -> dict:
    return {"city": city, "timestamp": OBSERVED_AT, "temperature": temperature}


async def test_quality_report_reports_the_registry_not_the_seed_tuple():
    features = FakeMongoCollection(
        [
            _feature("Madrid"),
            _feature("Barcelona"),
            _feature("Alicante"),  # rows, but not a registered station
        ]
    )
    db = FakeMongoDb(weather_features=features, cities=_registry("Barcelona", "Madrid"))

    report = await WeatherRepository(db).quality_report(days=1, now=OBSERVED_AT)

    assert [city.city for city in report] == ["Barcelona", "Madrid"]
    # One equality-prefixed query per registry station: 2, not the 14 seed
    # stations and not ``distinct("city")`` (which would add Alicante).
    assert len(features.find_calls) == 2
    assert {call[0]["city"] for call in features.find_calls} == {"Barcelona", "Madrid"}


async def test_station_deleted_from_the_registry_stops_being_reported():
    features = FakeMongoCollection([_feature("Madrid"), _feature("Barcelona")])
    registry = _registry("Madrid", "Barcelona")
    repo = WeatherRepository(FakeMongoDb(weather_features=features, cities=registry))

    before = await repo.quality_report(days=1, now=OBSERVED_AT)
    assert [city.city for city in before] == ["Barcelona", "Madrid"]

    # ``DELETE /cities/Barcelona``: the station leaves the registry, its
    # feature rows stay behind.
    registry.documents = [{"name": "Madrid", "name_key": "madrid"}]

    after = await repo.quality_report(days=1, now=OBSERVED_AT)
    assert [city.city for city in after] == ["Madrid"]
