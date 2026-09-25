"""Self-contained tests for ``/api/analytics/*`` (Batch 4, Contract 4).

The router is exercised through a stub ``AnalyticsRepository`` (its own fixture,
never the shared ``conftest``/``fakes``) plus a ``get_db`` override, so the suite
runs fully offline. The repository's own aggregation helpers and one end-to-end
join are covered with a tiny in-file Mongo double.
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
    _error_stats,
    _pearson,
    _wind_sector,
    get_analytics_repo,
)
from app.schemas.analytics import (
    ClimatologyPoint,
    ClimatologyResponse,
    CorrelationResponse,
    DailyPoint,
    DailyResponse,
    DiurnalCell,
    DiurnalResponse,
    ErrorByHourPoint,
    ErrorByHourResponse,
    WindRoseResponse,
    WindSector,
)

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures: stub repository + app/client, self-contained
# ---------------------------------------------------------------------------


class StubAnalyticsRepository:
    """Returns canned responses and records how the router called it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.daily_response: DailyResponse | None = None
        self.climatology_response: ClimatologyResponse | None = None
        self.wind_rose_response: WindRoseResponse | None = None
        self.diurnal_response: DiurnalResponse | None = None
        self.correlation_response: CorrelationResponse | None = None
        self.error_by_hour_response: ErrorByHourResponse | None = None
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

    async def daily(self, city: str, days: int, now: datetime) -> DailyResponse:
        self.calls.append(("daily", city, days))
        await self._gate()
        assert self.daily_response is not None
        return self.daily_response

    async def climatology(self, city: str, now: datetime) -> ClimatologyResponse:
        self.calls.append(("climatology", city))
        await self._gate()
        assert self.climatology_response is not None
        return self.climatology_response

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

    async def correlation(self, days: int, var: str, now: datetime) -> CorrelationResponse:
        self.calls.append(("correlation", days, var))
        await self._gate()
        assert self.correlation_response is not None
        return self.correlation_response

    async def error_by_hour(self, days: int, now: datetime) -> ErrorByHourResponse:
        self.calls.append(("error_by_hour", days))
        await self._gate()
        assert self.error_by_hour_response is not None
        return self.error_by_hour_response


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


def _daily_response() -> DailyResponse:
    return DailyResponse(
        city="Madrid",
        days=90,
        generated_at=NOW,
        points=[
            DailyPoint(
                date="2026-09-24",
                tmin=14.0,
                tmax=36.0,
                tmean=25.0,
                hdd=0.0,
                cdd=7.0,
                anomaly=1.5,
                heatwave=True,
            ),
            DailyPoint(date="2026-09-25"),  # no observations: every metric null
        ],
    )


# ---------------------------------------------------------------------------
# Happy path + null tolerance + caching, per endpoint
# ---------------------------------------------------------------------------


async def test_daily_returns_points_and_caches(client, analytics_repo):
    analytics_repo.daily_response = _daily_response()

    first = await client.get("/api/analytics/daily", params={"city": "Madrid", "days": 90})

    assert first.status_code == 200
    body = first.json()
    assert body["city"] == "Madrid"
    assert body["days"] == 90
    assert body["generated_at"].endswith("Z")
    assert set(body["points"][0]) == {
        "date",
        "tmin",
        "tmax",
        "tmean",
        "hdd",
        "cdd",
        "anomaly",
        "heatwave",
    }
    assert body["points"][0]["tmax"] == 36.0
    assert body["points"][0]["heatwave"] is True
    assert body["points"][1]["tmean"] is None
    assert body["points"][1]["heatwave"] is False
    assert "max-age=300" in first.headers["cache-control"]

    second = await client.get(
        "/api/analytics/daily",
        params={"city": "Madrid", "days": 90},
        headers={"If-None-Match": first.headers["etag"]},
    )
    assert second.status_code == 304
    assert analytics_repo.calls.count(("daily", "Madrid", 90)) == 1


async def test_climatology_returns_full_year_series(client, analytics_repo):
    series = [ClimatologyPoint(day_of_year=doy) for doy in range(1, 367)]
    series[264] = ClimatologyPoint(day_of_year=265, tmean=21.0, tmin=12.0, tmax=30.0, n=5)
    analytics_repo.climatology_response = ClimatologyResponse(
        city="Madrid",
        generated_at=NOW,
        basis_years=2.0,
        series=series,
    )

    response = await client.get("/api/analytics/climatology", params={"city": "Madrid"})

    assert response.status_code == 200
    body = response.json()
    assert body["basis_years"] == 2.0
    assert len(body["series"]) == 366
    assert body["series"][0] == {
        "day_of_year": 1,
        "tmean": None,
        "tmin": None,
        "tmax": None,
        "n": 0,
    }
    assert body["series"][264] == {
        "day_of_year": 265,
        "tmean": 21.0,
        "tmin": 12.0,
        "tmax": 30.0,
        "n": 5,
    }


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


async def test_correlation_returns_matrix_with_nulls(client, analytics_repo):
    analytics_repo.correlation_response = CorrelationResponse(
        days=90,
        var="temperature",
        generated_at=NOW,
        cities=["Barcelona", "Madrid"],
        matrix=[[1.0, None], [None, None]],
    )

    response = await client.get(
        "/api/analytics/correlation", params={"days": 90, "var": "temperature"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["var"] == "temperature"
    assert body["cities"] == ["Barcelona", "Madrid"]
    assert body["matrix"] == [[1.0, None], [None, None]]


async def test_error_by_hour_returns_points_with_nulls(client, analytics_repo):
    analytics_repo.error_by_hour_response = ErrorByHourResponse(
        days=30,
        generated_at=NOW,
        points=[
            ErrorByHourPoint(
                horizon_hours=1, hour=13, n=2, mae=1.0, rmse=1.0, bias=-0.5, persistence_mae=2.0
            ),
            ErrorByHourPoint(horizon_hours=3, hour=15),
        ],
    )

    response = await client.get("/api/analytics/error-by-hour", params={"days": 30})

    assert response.status_code == 200
    body = response.json()
    assert body["points"][0] == {
        "horizon_hours": 1,
        "hour": 13,
        "n": 2,
        "mae": 1.0,
        "rmse": 1.0,
        "bias": -0.5,
        "persistence_mae": 2.0,
    }
    assert body["points"][1]["mae"] is None
    assert body["points"][1]["n"] == 0


async def test_openapi_documents_the_six_analytics_paths(client):
    schema = (await client.get("/openapi.json")).json()
    assert {
        "/api/analytics/daily",
        "/api/analytics/climatology",
        "/api/analytics/wind-rose",
        "/api/analytics/diurnal",
        "/api/analytics/correlation",
        "/api/analytics/error-by-hour",
    } <= set(schema["paths"])


async def test_analytics_bounds_concurrent_repo_computation(client, analytics_repo):
    analytics_repo.daily_response = _daily_response()
    analytics_repo.climatology_response = ClimatologyResponse(
        city="Madrid", generated_at=NOW, basis_years=1.0, series=[]
    )
    analytics_repo.wind_rose_response = WindRoseResponse(
        city="Madrid", days=90, generated_at=NOW, sectors=[]
    )
    # Event-based gating: release only once two repo calls are provably in
    # flight. A sleep here would be a race on a loaded CI runner.
    analytics_repo.hold = asyncio.Event()

    # Three distinct cache keys (no single-flight coalescing): the module-level
    # semaphore is the only thing capping simultaneous repo work.
    requests = asyncio.gather(
        client.get("/api/analytics/daily", params={"city": "Madrid", "days": 90}),
        client.get("/api/analytics/climatology", params={"city": "Madrid"}),
        client.get("/api/analytics/wind-rose", params={"city": "Madrid", "days": 90}),
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


@pytest.mark.parametrize("days", [6, 181])
async def test_daily_rejects_out_of_range_days(client, days):
    response = await client.get("/api/analytics/daily", params={"city": "Madrid", "days": days})
    assert response.status_code == 422


async def test_daily_rejects_missing_or_empty_city(client):
    assert (await client.get("/api/analytics/daily", params={"days": 90})).status_code == 422
    assert (
        await client.get("/api/analytics/daily", params={"city": "", "days": 90})
    ).status_code == 422


@pytest.mark.parametrize("days", [7, 180])
async def test_daily_accepts_boundary_days(client, analytics_repo, days):
    analytics_repo.daily_response = _daily_response()
    response = await client.get("/api/analytics/daily", params={"city": "Madrid", "days": days})
    assert response.status_code == 200


async def test_climatology_requires_city(client):
    assert (await client.get("/api/analytics/climatology")).status_code == 422


@pytest.mark.parametrize("days", [6, 366])
async def test_wind_rose_rejects_out_of_range_days(client, days):
    response = await client.get("/api/analytics/wind-rose", params={"city": "Madrid", "days": days})
    assert response.status_code == 422


async def test_wind_rose_requires_city(client):
    assert (await client.get("/api/analytics/wind-rose")).status_code == 422


@pytest.mark.parametrize("days", [6, 366])
async def test_diurnal_rejects_out_of_range_days(client, days):
    assert (await client.get("/api/analytics/diurnal", params={"days": days})).status_code == 422


@pytest.mark.parametrize("days", [6, 366])
async def test_correlation_rejects_out_of_range_days(client, days):
    assert (
        await client.get("/api/analytics/correlation", params={"days": days})
    ).status_code == 422


async def test_correlation_rejects_unknown_var(client):
    response = await client.get("/api/analytics/correlation", params={"var": "bogus"})
    assert response.status_code == 422


@pytest.mark.parametrize("days", [6, 91])
async def test_error_by_hour_rejects_out_of_range_days(client, days):
    assert (
        await client.get("/api/analytics/error-by-hour", params={"days": days})
    ).status_code == 422


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------


def test_pearson_known_and_guards():
    assert _pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert _pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == -1.0
    # fewer than 3 shared days
    assert _pearson([1.0, 2.0], [2.0, 4.0]) is None
    # zero variance is undefined
    assert _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_wind_sector_maps_and_handles_missing():
    assert _wind_sector(0.0) == 0
    assert _wind_sector(90.0) == 4
    assert _wind_sector(180.0) == 8
    assert _wind_sector(270.0) == 12
    assert _wind_sector(360.0) == 0
    assert _wind_sector(None) is None


def test_error_stats():
    mae, rmse, bias = _error_stats([1.0, -2.0])
    assert mae == 1.5
    assert rmse == pytest.approx((2.5) ** 0.5)
    assert bias == -0.5
    assert _error_stats([]) == (None, None, None)


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
    def __init__(self, features: list[dict], predictions: list[dict]) -> None:
        self._collections = {
            "weather_features": _Collection(features),
            "weather_predictions": _Collection(predictions),
        }

    def __getitem__(self, name: str) -> _Collection:
        return self._collections[name]


async def test_repository_daily_computes_degree_days_anomaly_and_heatwave():
    def row(year: int, month: int, day: int, temperature: float | None) -> dict:
        return {
            "city": "Madrid",
            "timestamp": datetime(year, month, day, 12, 0, tzinfo=timezone.utc),
            "temperature": temperature,
        }

    features = [
        row(2026, 9, 22, 20.0),
        row(2026, 9, 22, 24.0),
        row(2025, 9, 23, 18.0),  # two years for this day-of-year -> anomaly
        row(2026, 9, 23, 30.0),
        row(2026, 9, 23, 36.0),  # heatwave day 1
        row(2026, 9, 24, 34.0),
        row(2026, 9, 24, 38.0),  # heatwave day 2
        row(2026, 9, 25, 33.0),
        row(2026, 9, 25, 37.0),  # heatwave day 3
        row(2026, 9, 25, None),  # null observation is ignored, never fabricated
    ]
    repo = AnalyticsRepository(_Db(features, []))

    result = await repo.daily("Madrid", 4, NOW)

    assert [point.date for point in result.points] == [
        "2026-09-22",
        "2026-09-23",
        "2026-09-24",
        "2026-09-25",
    ]
    first = result.points[0]
    assert (first.tmin, first.tmax, first.tmean) == (20.0, 24.0, 22.0)
    assert first.hdd == 0.0 and first.cdd == 4.0
    assert first.anomaly is None  # only one year contributes to 09-22
    assert first.heatwave is False
    assert [point.heatwave for point in result.points[1:]] == [True, True, True]
    assert result.points[1].anomaly is not None


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
    repo = AnalyticsRepository(_Db(features, []))

    result = await repo.wind_rose("Madrid", 90, NOW)

    assert len(result.sectors) == 16
    assert result.sectors[0].count == 1 and result.sectors[0].mean_speed == 3.6
    assert result.sectors[4].mean_speed == 3.6
    assert result.sectors[8].mean_speed == 3.6
    assert result.sectors[12].mean_speed == 3.6
    assert sum(sector.count for sector in result.sectors) == 4  # null direction skipped
    assert result.sectors[1].mean_speed is None


async def test_repository_error_by_hour_joins_target_and_excludes_missing():
    def feature(hour: int, temperature: float) -> dict:
        return {
            "city": "Madrid",
            "timestamp": datetime(2026, 9, 25, hour, 0, tzinfo=timezone.utc),
            "temperature": temperature,
            "hour": (hour + 2) % 24,  # arbitrary local hour, read back verbatim
        }

    def prediction(
        source_hour: int,
        horizon: int,
        predicted: float | None,
        observed: float | None,
    ) -> dict:
        return {
            "city": "Madrid",
            "source_timestamp": datetime(2026, 9, 25, source_hour, 0, tzinfo=timezone.utc),
            "prediction_timestamp": NOW,
            "horizon_hours": horizon,
            "predicted_temperature": predicted,
            "observed_temperature": observed,
        }

    features = [feature(10, 20.0), feature(11, 22.0), feature(12, 24.0), feature(13, 26.0)]
    predictions = [
        prediction(10, 1, 21.0, 20.0),  # target 11:00 -> error -1, persistence 2
        prediction(10, 2, 25.0, 20.0),  # target 12:00 -> error 1, persistence 4
        prediction(13, 1, 99.0, 26.0),  # target 14:00 has no observation -> excluded
        prediction(10, 1, None, 20.0),  # no prediction -> excluded
        prediction(11, 1, 23.0, None),  # target 12:00 -> error -1, no persistence
    ]
    repo = AnalyticsRepository(_Db(features, predictions))

    result = await repo.error_by_hour(30, NOW)

    points = {(point.horizon_hours, point.hour): point for point in result.points}
    assert set(points) == {(1, 13), (2, 14), (1, 14)}
    assert points[(1, 13)].n == 1
    assert points[(1, 13)].mae == 1.0
    assert points[(1, 13)].bias == -1.0
    assert points[(1, 13)].persistence_mae == 2.0
    assert points[(2, 14)].bias == 1.0
    assert points[(2, 14)].persistence_mae == 4.0
    assert points[(1, 14)].persistence_mae is None


async def test_repository_error_by_hour_buckets_minute_aligned_observations():
    """Feature timestamps are not minute-aligned (ingest runs at :50); the join
    must bucket both sides to the hour or it drops every row."""
    features = [
        {
            "city": "Madrid",
            "timestamp": datetime(2026, 9, 25, 12, 50, tzinfo=timezone.utc),
            "temperature": 23.0,
            "hour": 14,  # local hour, read back verbatim
        },
    ]
    predictions = [
        # source 11:00 + 1h -> target 12:00, which the :50 feature buckets into.
        {
            "city": "Madrid",
            "source_timestamp": datetime(2026, 9, 25, 11, 0, tzinfo=timezone.utc),
            "prediction_timestamp": NOW,
            "horizon_hours": 1,
            "predicted_temperature": 22.0,
            "observed_temperature": 20.0,
        },
        # target 13:00 has no observation -> excluded.
        {
            "city": "Madrid",
            "source_timestamp": datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc),
            "prediction_timestamp": NOW,
            "horizon_hours": 1,
            "predicted_temperature": 25.0,
            "observed_temperature": 24.0,
        },
    ]
    repo = AnalyticsRepository(_Db(features, predictions))

    result = await repo.error_by_hour(30, NOW)

    assert len(result.points) == 1
    point = result.points[0]
    assert (point.horizon_hours, point.hour) == (1, 14)
    assert point.n == 1
    assert point.mae == pytest.approx(1.0)  # |22 - 23|
    assert point.bias == pytest.approx(-1.0)
    assert point.persistence_mae == pytest.approx(3.0)  # |23 - 20|
