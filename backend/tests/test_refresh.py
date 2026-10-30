"""``POST /refresh``: fetch, document shape, rate limits, failure modes (offline)."""

import ast
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.rate_limit import RefreshRateLimiter
from app.main import create_app
from app.repositories.city_repo import get_city_repo
from app.repositories.weather_repo import WeatherRepository, get_weather_repo
from app.services.openmeteo import (
    CURRENT_VARIABLES,
    OPEN_METEO_URL,
    OpenMeteoService,
    build_document,
    get_openmeteo_service,
)
from tests.fakes import (
    OPEN_METEO_TIME,
    FakeMongoCollection,
    FakeMongoDb,
    open_meteo_payload,
)

UTC = timezone.utc
EXPECTED_TIMESTAMP = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


def _service_with(handler) -> OpenMeteoService:
    return OpenMeteoService(transport=httpx.MockTransport(handler))


def _override(application, service) -> None:
    application.dependency_overrides[get_openmeteo_service] = lambda: service


def _reset_limiter(application, **bounds) -> None:
    application.state.refresh_limiter = RefreshRateLimiter(
        cooldown_seconds=bounds.pop("cooldown_seconds", 900),
        max_per_minute=bounds.pop("max_per_minute", 6),
        max_per_day=bounds.pop("max_per_day", 48),
        **bounds,
    )


# ---------------------------------------------------------------------------
# Document shape / repository write
# ---------------------------------------------------------------------------


def test_refresh_settings_defaults():
    settings = Settings(_env_file=None)

    assert settings.refresh_enabled is False
    assert settings.refresh_city_cooldown_seconds == 900
    assert settings.refresh_max_per_minute == 6
    assert settings.refresh_max_per_day == 48


def _ingest_script_constant(name: str):
    """Read a module-level literal from the ingest script without importing it.

    Importing the script would drag in ``requests`` (a pipeline dependency the
    backend image does not install) and execute it; ``ast`` keeps this a pure
    source-parity check.
    """
    path = Path(__file__).parents[2] / "scripts" / "ingest_weather.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level literal in {path}")


def test_current_variables_match_the_ingest_script():
    assert _ingest_script_constant("CURRENT_VARIABLES") == CURRENT_VARIABLES
    assert _ingest_script_constant("OPEN_METEO_URL") == OPEN_METEO_URL


def test_build_document_mirrors_the_ingest_shape():
    payload = {**open_meteo_payload(), "city": "Madrid"}

    document = build_document(payload)

    assert document is not None
    assert document["_id"] == f"Madrid_{OPEN_METEO_TIME}"
    assert document["city"] == "Madrid"
    assert document["timestamp"] == EXPECTED_TIMESTAMP
    assert document["payload"] == payload
    assert isinstance(document["fetched_at"], datetime)
    assert document["fetched_at"].tzinfo is not None


def test_build_document_rejects_a_payload_without_current_time():
    assert build_document({"city": "Madrid", "current": {}}) is None
    assert build_document({"city": "Madrid"} | {"current": {"time": None}}) is None


async def test_repository_upserts_on_the_same_id():
    raw = FakeMongoCollection()
    repo = WeatherRepository(FakeMongoDb(raw_weather=raw))
    first = build_document({**open_meteo_payload(), "city": "Madrid"})
    second = build_document({**open_meteo_payload(temperature=25.0), "city": "Madrid"})
    assert first is not None and second is not None

    await repo.save_raw_observation(first)
    await repo.save_raw_observation(second)

    assert len(raw.replace_calls) == 2
    for query, _document, upsert in raw.replace_calls:
        assert query == {"_id": f"Madrid_{OPEN_METEO_TIME}"}
        assert upsert is True
    # Same observation time -> one row, replaced rather than duplicated.
    assert len(raw.documents) == 1
    assert raw.documents[0]["payload"]["current"]["temperature_2m"] == 25.0


# ---------------------------------------------------------------------------
# Happy path over the real service + MockTransport
# ---------------------------------------------------------------------------


async def test_refresh_writes_the_expected_document(client, weather_repo):
    response = await client.post("/api/v1/refresh", json={"city": "Madrid"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "city",
        "source_timestamp",
        "fetched_at",
        "features_refreshed",
        "predictions_refreshed",
    }
    assert body["city"] == "Madrid"
    assert body["source_timestamp"] == "2026-09-23T12:00:00Z"
    assert body["fetched_at"].endswith("Z")
    assert body["features_refreshed"] is False
    assert body["predictions_refreshed"] is False

    assert weather_repo.calls == ["save_raw_observation"]
    (document,) = weather_repo.saved_raw
    assert document["_id"] == f"Madrid_{OPEN_METEO_TIME}"
    assert document["city"] == "Madrid"
    assert document["timestamp"] == EXPECTED_TIMESTAMP
    # The payload is the untouched upstream response plus the city tag.
    assert document["payload"] == {**open_meteo_payload(), "city": "Madrid"}
    assert set(document["payload"]["current"]) >= {
        "temperature_2m",
        "wind_speed_10m",
        "surface_pressure",
        "weather_code",
    }


async def test_refresh_calls_open_meteo_with_the_ingest_params(client, app, weather_repo):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=open_meteo_payload())

    _override(app, _service_with(handler))

    response = await client.post("/api/refresh", json={"city": "Madrid"})

    assert response.status_code == 200
    (request,) = seen
    assert str(request.url.copy_with(query=None)) == OPEN_METEO_URL
    assert request.url.params["wind_speed_unit"] == "ms"
    assert request.url.params["timeformat"] == "unixtime"
    assert request.url.params["timezone"] == "auto"
    assert request.url.params["latitude"] == "40.4168"
    assert request.url.params["longitude"] == "-3.7038"
    assert request.url.params.get_list("current") == CURRENT_VARIABLES


async def test_both_prefixes_answer(client, app):
    _reset_limiter(app, cooldown_seconds=0)

    for prefix in ("/api", "/api/v1"):
        response = await client.post(f"{prefix}/refresh", json={"city": "Madrid"})
        assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Validation and unknown city
# ---------------------------------------------------------------------------


async def test_unknown_city_is_404_and_echoes_nothing(client, weather_repo):
    response = await client.post("/api/refresh", json={"city": "SECRET-abc123"})

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "city_not_found"
    assert error["request_id"] == response.headers["x-request-id"]
    assert "SECRET-abc123" not in response.text
    assert weather_repo.calls == []


async def test_city_match_is_case_sensitive(client, weather_repo):
    response = await client.post("/api/refresh", json={"city": "madrid"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "city_not_found"
    assert weather_repo.calls == []


async def test_overlong_city_is_a_validation_error_without_the_submitted_value(client):
    response = await client.post("/api/refresh", json={"city": "x" * 101})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"] == response.headers["x-request-id"]
    assert "x" * 101 not in response.text


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


async def test_second_call_within_the_cooldown_is_429_with_retry_after(client):
    assert (await client.post("/api/refresh", json={"city": "Madrid"})).status_code == 200

    response = await client.post("/api/refresh", json={"city": "Madrid"})

    assert response.status_code == 429
    error = response.json()["error"]
    assert error["code"] == "refresh_rate_limited"
    assert error["request_id"] == response.headers["x-request-id"]
    retry_after = int(response.headers["retry-after"])
    assert 1 <= retry_after <= 900


async def test_per_minute_cap(client, app):
    _reset_limiter(app, cooldown_seconds=0, max_per_minute=1)

    first = await client.post("/api/refresh", json={"city": "Madrid"})
    second = await client.post("/api/refresh", json={"city": "Alicante"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "refresh_rate_limited"
    assert 1 <= int(second.headers["retry-after"]) <= 60


async def test_per_day_cap(client, app):
    _reset_limiter(app, cooldown_seconds=0, max_per_minute=10, max_per_day=1)

    first = await client.post("/api/refresh", json={"city": "Madrid"})
    second = await client.post("/api/refresh", json={"city": "Alicante"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert 1 <= int(second.headers["retry-after"]) <= 86400


async def test_limiter_accepts_only_the_cap_under_concurrency():
    clock = [1000.0]
    limiter = RefreshRateLimiter(
        cooldown_seconds=0,
        max_per_minute=3,
        max_per_day=10,
        clock=lambda: clock[0],
    )

    decisions = await asyncio.gather(*(limiter.acquire(f"city-{index}") for index in range(10)))

    assert sum(1 for decision in decisions if decision.allowed) == 3
    assert all(decision.retry_after >= 1 for decision in decisions if not decision.allowed)


def test_limiter_windows_roll_over_with_the_clock():
    clock = [0.0]
    limiter = RefreshRateLimiter(
        cooldown_seconds=900,
        max_per_minute=1,
        max_per_day=2,
        clock=lambda: clock[0],
    )

    async def scenario() -> None:
        assert (await limiter.acquire("Madrid")).allowed is True
        # Cooldown still holds at +899 s, and the minute window has rolled.
        clock[0] = 899.0
        blocked = await limiter.acquire("Madrid")
        assert blocked.allowed is False
        assert blocked.retry_after == 1
        # At +900 s both the cooldown and the minute window are clear.
        clock[0] = 900.0
        assert (await limiter.acquire("Madrid")).allowed is True
        # The daily cap (2) is now spent even though a minute has passed.
        clock[0] = 1000.0
        assert (await limiter.acquire("Madrid")).allowed is False
        # A new day resets the daily counter.
        clock[0] = 900.0 + 86400.0
        assert (await limiter.acquire("Madrid")).allowed is True

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Disabled and upstream failures
# ---------------------------------------------------------------------------


async def test_default_config_is_503_without_touching_upstream_or_storage(weather_repo, city_repo):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=open_meteo_payload())

    application = create_app(Settings(_env_file=None))
    application.dependency_overrides[get_weather_repo] = lambda: weather_repo
    application.dependency_overrides[get_city_repo] = lambda: city_repo
    application.dependency_overrides[get_openmeteo_service] = lambda: _service_with(handler)

    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test"
    ) as default_client:
        response = await default_client.post("/api/v1/refresh", json={"city": "Madrid"})

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "refresh_disabled"
    assert error["message"] == "The refresh endpoint is disabled"
    assert error["request_id"] == response.headers["x-request-id"]
    assert seen == []
    assert weather_repo.calls == []
    assert city_repo.calls == []


async def test_disabled_is_503(client, app, settings, weather_repo):
    app.state.settings = settings.model_copy(update={"refresh_enabled": False})

    response = await client.post("/api/refresh", json={"city": "Madrid"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "refresh_disabled"
    assert weather_repo.calls == []


async def test_upstream_500_is_502(client, app, weather_repo):
    _override(app, _service_with(lambda _request: httpx.Response(500, text="boom")))

    response = await client.post("/api/refresh", json={"city": "Madrid"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "refresh_upstream_error"
    assert "boom" not in response.text
    assert weather_repo.calls == []


async def test_unusable_payload_is_502(client, app, weather_repo):
    _override(app, _service_with(lambda _request: httpx.Response(200, json={"latitude": 40.4})))

    response = await client.post("/api/refresh", json={"city": "Madrid"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "refresh_upstream_error"
    assert weather_repo.calls == []


async def test_non_object_payload_is_502(client, app, weather_repo):
    _override(app, _service_with(lambda _request: httpx.Response(200, json=[1, 2, 3])))

    response = await client.post("/api/refresh", json={"city": "Madrid"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "refresh_upstream_error"
    assert weather_repo.calls == []


async def test_success_response_carries_the_request_id(client):
    response = await client.post(
        "/api/refresh",
        json={"city": "Madrid"},
        headers={"x-request-id": "refresh-test-1"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "refresh-test-1"
