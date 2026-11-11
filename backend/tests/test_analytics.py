"""Self-contained tests for ``/api/analytics/*`` (Batch 4, Contract 4).

The router is exercised through a stub ``AnalyticsRepository`` (its own fixture,
never the shared ``conftest``/``fakes``) plus a ``get_db`` override, so the suite
runs fully offline. The repository's own aggregation helpers and the diurnal
registry-driven aggregate are covered with a tiny in-file Mongo double.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.db.mongo import get_db
from app.main import create_app
from app.repositories.analytics_repo import (
    AnalyticsRepository,
    _wind_sector,
    get_analytics_repo,
)
from app.schemas.analytics import (
    DiurnalCell,
    DiurnalResponse,
    WindRoseResponse,
    WindSector,
)
from tests.fakes import FakeMongoCollection, FakeMongoDb

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures: stub repository + app/client, self-contained
# ---------------------------------------------------------------------------


class StubAnalyticsRepository:
    """Returns canned responses and records how the router called it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.wind_rose_response: WindRoseResponse | None = None
        self.diurnal_response: DiurnalResponse | None = None
        #: Optional per-call delay + live concurrency, to prove the router's
        #: compute semaphore caps simultaneous repo work.
        self.delay = 0.0
        #: When set, repo calls park on it (instead of sleeping) so a test can
        #: assert real overlap without a wall-clock race.
        self.hold: asyncio.Event | None = None
        self.reached_two = asyncio.Event()
        self.active = 0
        self.max_concurrent = 0

    async def _gate(self) -> None:
        self.active += 1
        self.max_concurrent = max(self.max_concurrent, self.active)
        try:
            if self.hold is not None:
                if self.active >= 2:
                    self.reached_two.set()
                await self.hold.wait()
            elif self.delay:
                await asyncio.sleep(self.delay)
        finally:
            self.active -= 1

    async def wind_rose(self, city: str, days: int, now: datetime) -> WindRoseResponse:
        self.calls.append(("wind_rose", city, days))
        await self._gate()
        assert self.wind_rose_response is not None
        return self.wind_rose_response

    async def diurnal(self, days: int, now: datetime) -> DiurnalResponse:
        self.calls.append(("diurnal", days))
        await self._gate()
        assert self.diurnal_response is not None
        return self.diurnal_response


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        mongo_uri="mongodb://localhost:27017",
        cors_origins="http://localhost:5173",
    )


@pytest.fixture
def analytics_repo() -> StubAnalyticsRepository:
    return StubAnalyticsRepository()


@pytest.fixture
def app(settings, analytics_repo):
    application = create_app(settings)
    application.dependency_overrides[get_analytics_repo] = lambda: analytics_repo
    application.dependency_overrides[get_db] = lambda: object()
    return application


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


# ---------------------------------------------------------------------------
# Happy path + null tolerance + caching, per endpoint
# ---------------------------------------------------------------------------


async def test_wind_rose_returns_16_sectors_and_null_speed(client, analytics_repo):
    analytics_repo.wind_rose_response = WindRoseResponse(
        city="Madrid",
        days=90,
        generated_at=NOW,
        sectors=[WindSector(sector=index) for index in range(16)],
    )
    analytics_repo.wind_rose_response.sectors[0] = WindSector(sector=0, count=3, mean_speed=18.0)

    response = await client.get("/api/analytics/wind-rose", params={"city": "Madrid", "days": 90})

    assert response.status_code == 200
    body = response.json()
    assert len(body["sectors"]) == 16
    assert body["sectors"][0] == {"sector": 0, "count": 3, "mean_speed": 18.0}
    assert body["sectors"][1] == {"sector": 1, "count": 0, "mean_speed": None}


async def test_diurnal_returns_cells_and_null_city(client, analytics_repo):
    analytics_repo.diurnal_response = DiurnalResponse(
        days=90,
        generated_at=NOW,
        cells=[
            DiurnalCell(city="Madrid", hour=0, tmean=11.0, n=4),
            DiurnalCell(city="Madrid", hour=1),
        ],
    )

    response = await client.get("/api/analytics/diurnal", params={"days": 90})

    assert response.status_code == 200
    body = response.json()
    assert body["cells"][0] == {"city": "Madrid", "hour": 0, "tmean": 11.0, "n": 4}
    assert body["cells"][1] == {"city": "Madrid", "hour": 1, "tmean": None, "n": 0}


async def test_diurnal_is_cached_with_etag_and_304(client, analytics_repo):
    analytics_repo.diurnal_response = DiurnalResponse(
        days=90,
        generated_at=NOW,
        cells=[DiurnalCell(city="Madrid", hour=12, tmean=25.0, n=4)],
    )

    first = await client.get("/api/analytics/diurnal", params={"days": 90})

    assert first.status_code == 200
    assert first.json()["generated_at"].endswith("Z")
    assert "max-age=300" in first.headers["cache-control"]

    second = await client.get(
        "/api/analytics/diurnal",
        params={"days": 90},
        headers={"If-None-Match": first.headers["etag"]},
    )
    assert second.status_code == 304
    assert analytics_repo.calls.count(("diurnal", 90)) == 1


async def test_openapi_documents_the_two_analytics_paths(client):
    schema = (await client.get("/openapi.json")).json()
    assert {
        "/api/analytics/wind-rose",
        "/api/analytics/diurnal",
    } <= set(schema["paths"])


async def test_openapi_omits_the_retired_analytics_paths(client):
    """Zero-caller endpoints stay deleted (audit finding 13)."""
    schema = (await client.get("/openapi.json")).json()
    retired = {
        "/api/analytics/daily",
        "/api/analytics/climatology",
        "/api/analytics/correlation",
        "/api/analytics/error-by-hour",
    }
    assert retired & set(schema["paths"]) == set()


async def test_analytics_bounds_concurrent_repo_computation(client, analytics_repo):
    analytics_repo.wind_rose_response = WindRoseResponse(
        city="Madrid", days=90, generated_at=NOW, sectors=[]
    )
    # Event-based gating: release only once two repo calls are provably in
    # flight. A sleep here would be a race on a loaded CI runner.
    analytics_repo.hold = asyncio.Event()

    # Three distinct cache keys (no single-flight coalescing): the module-level
    # semaphore is the only thing capping simultaneous repo work.
    requests = asyncio.gather(
        client.get("/api/analytics/wind-rose", params={"city": "Madrid", "days": 30}),
        client.get("/api/analytics/wind-rose", params={"city": "Madrid", "days": 90}),
        client.get("/api/analytics/wind-rose", params={"city": "Madrid", "days": 180}),
    )
    await asyncio.wait_for(analytics_repo.reached_two.wait(), timeout=2.0)
    assert analytics_repo.max_concurrent == 2

    analytics_repo.hold.set()
    responses = await requests

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert analytics_repo.max_concurrent == 2


# ---------------------------------------------------------------------------
# Parameter validation, per endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("days", [6, 366])
async def test_wind_rose_rejects_out_of_range_days(client, days):
    response = await client.get("/api/analytics/wind-rose", params={"city": "Madrid", "days": days})
    assert response.status_code == 422


async def test_wind_rose_requires_city(client):
    assert (await client.get("/api/analytics/wind-rose")).status_code == 422


@pytest.mark.parametrize("days", [6, 366])
async def test_diurnal_rejects_out_of_range_days(client, days):
    assert (await client.get("/api/analytics/diurnal", params={"days": days})).status_code == 422


@pytest.mark.parametrize("days", [7, 365])
async def test_diurnal_accepts_boundary_days(client, analytics_repo, days):
    analytics_repo.diurnal_response = DiurnalResponse(days=days, generated_at=NOW, cells=[])
    response = await client.get("/api/analytics/diurnal", params={"days": days})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------


def test_wind_sector_maps_and_handles_missing():
    assert _wind_sector(0.0) == 0
    assert _wind_sector(90.0) == 4
    assert _wind_sector(180.0) == 8
    assert _wind_sector(270.0) == 12
    assert _wind_sector(360.0) == 0
    assert _wind_sector(None) is None


# ---------------------------------------------------------------------------
# Repository end-to-end (tiny in-file Mongo double)
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self._documents = documents

    def sort(self, spec, direction=None):  # noqa: ANN001, ANN201 - duck-typed
        return self

    def __aiter__(self):
        self._iterator = iter(self._documents)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None


def _matches(document: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, expected in query.items():
        actual = document.get(key)
        if isinstance(expected, dict):
            if "$gte" in expected and not (actual is not None and actual >= expected["$gte"]):
                return False
            if "$lte" in expected and not (actual is not None and actual <= expected["$lte"]):
                return False
        elif actual != expected:
            return False
    return True


class _Collection:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def find(self, query, projection=None):  # noqa: ANN001, ANN201 - duck-typed
        return _Cursor([doc for doc in self.documents if _matches(doc, query)])


class _Db:
    def __init__(self, features: list[dict]) -> None:
        self._collections = {
            "weather_features": _Collection(features),
            "cities": _Collection([]),  # unseeded: the repositories use the seed list
        }

    def __getitem__(self, name: str) -> _Collection:
        return self._collections[name]


async def test_repository_wind_rose_buckets_and_converts_ms_to_kmh():
    features = [
        {
            "city": "Madrid",
            "timestamp": datetime(2026, 9, 20, hour, tzinfo=timezone.utc),
            "wind_direction": direction,
            "wind_speed": 1.0,  # m/s
        }
        for hour, direction in enumerate((0.0, 90.0, 180.0, 270.0, None))
    ]
    repo = AnalyticsRepository(_Db(features))

    result = await repo.wind_rose("Madrid", 90, NOW)

    assert len(result.sectors) == 16
    assert result.sectors[0].count == 1 and result.sectors[0].mean_speed == 3.6
    assert result.sectors[4].mean_speed == 3.6
    assert result.sectors[8].mean_speed == 3.6
    assert result.sectors[12].mean_speed == 3.6
    assert sum(sector.count for sector in result.sectors) == 4  # null direction skipped
    assert result.sectors[1].mean_speed is None


# ---------------------------------------------------------------------------
# The aggregate city set is the Mongo registry, not a code constant (finding 4)
# ---------------------------------------------------------------------------


def _registry(*names: str) -> FakeMongoCollection:
    """A ``cities`` collection holding just these stations."""
    return FakeMongoCollection([{"name": name, "name_key": name.casefold()} for name in names])


def _hourly(city: str, hour: int, temperature: float) -> dict:
    return {
        "city": city,
        "timestamp": datetime(2026, 9, 25, hour, 0, tzinfo=timezone.utc),
        "hour": hour,
        "temperature": temperature,
    }


async def test_diurnal_covers_the_registry_not_the_seed_list():
    features = FakeMongoCollection(
        [
            _hourly("Madrid", 12, 25.0),
            _hourly("Barcelona", 12, 22.0),
            _hourly("Alicante", 12, 30.0),  # rows, but not a registered station
        ]
    )
    registry = _registry("Madrid", "Barcelona")
    db = FakeMongoDb(weather_features=features, cities=registry)

    result = await AnalyticsRepository(db).diurnal(1, NOW)

    assert sorted({cell.city for cell in result.cells}) == ["Barcelona", "Madrid"]
    assert len(result.cells) == 2 * 24
    # The registry is read once per aggregate, then queried once per station.
    assert registry.find_calls == [({}, {"name": 1})]
    assert [call[0]["city"] for call in features.find_calls] == ["Barcelona", "Madrid"]


async def test_station_deleted_from_the_registry_stops_being_reported():
    features = FakeMongoCollection([_hourly("Madrid", 12, 25.0), _hourly("Barcelona", 12, 22.0)])
    registry = _registry("Madrid", "Barcelona")
    db = FakeMongoDb(weather_features=features, cities=registry)

    before = await AnalyticsRepository(db).diurnal(1, NOW)
    assert {cell.city for cell in before.cells} == {"Barcelona", "Madrid"}

    # ``DELETE /cities/Barcelona``: the station's rows stay, the station goes.
    registry.documents = [{"name": "Madrid", "name_key": "madrid"}]

    after = await AnalyticsRepository(db).diurnal(1, NOW)
    assert {cell.city for cell in after.cells} == {"Madrid"}
