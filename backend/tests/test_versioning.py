"""Batch-5 R6: ``/api/v1`` is canonical, ``/api`` is a working legacy alias.

Both prefixes are mounted from the *same* aggregate router, so the two surfaces
cannot drift; these tests pin that invariant (same endpoint object, same payload,
same validator) rather than spot-checking a few URLs.
"""

from datetime import datetime, timezone

import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from app.main import API_LEGACY_PREFIX, API_V1_PREFIX
from app.repositories.analytics_repo import get_analytics_repo
from app.schemas.analytics import DiurnalCell, DiurnalResponse

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

#: One representative endpoint per sub-router family (health, cities, weather,
#: models, analytics). ``cached`` marks the routes served through the TTL cache,
#: where both prefixes must also agree on the ETag.
CASES = [
    ("/health", None, False),
    ("/cities", None, True),
    ("/weather/current", None, True),
    ("/models", None, True),
    ("/analytics/diurnal", {"days": 30}, True),
]

#: ``contract``-style pairs used for the structural assertions below.
PAIRS = [("/api" + path, API_V1_PREFIX + path) for path, _, _ in CASES]


class StubAnalyticsRepository:
    """Just enough of the analytics repo for one cached route."""

    async def diurnal(self, days: int, now: datetime) -> DiurnalResponse:
        return DiurnalResponse(
            days=days,
            generated_at=NOW,
            cells=[DiurnalCell(city="Madrid", hour=12, tmean=22.0, n=3)],
        )


def _mounted_routes(app):
    """Yield ``(APIRoute, effective path)`` for every mounted route.

    FastAPI >= 0.138 includes routers lazily: ``app.routes`` holds
    ``_IncludedRouter`` placeholders instead of one ``APIRoute`` per mount, and
    ``route.path`` does not exist on them. ``effective_route_contexts()``
    resolves a placeholder into its original route objects plus the path each
    mount produces. Older FastAPI versions eagerly include plain ``APIRoute``s,
    so those are yielded directly.
    """
    router = app.router
    resolver = getattr(router, "effective_route_contexts", None)
    if resolver is not None:
        for context in resolver():
            yield context.original_route, context.path
        return
    for route in router.routes:
        resolver = getattr(route, "effective_route_contexts", None)
        if resolver is not None:
            for context in resolver():
                yield context.original_route, context.path
        elif isinstance(route, APIRoute):
            yield route, route.path


@pytest.fixture
def versioned_app(app):
    app.dependency_overrides[get_analytics_repo] = lambda: StubAnalyticsRepository()
    return app


@pytest.fixture
async def versioned_client(versioned_app):
    transport = ASGITransport(app=versioned_app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


def _comparable(path: str, body):
    """The liveness payload embeds the wall clock; everything else must match exactly."""
    if path == "/health":
        return {key: value for key, value in body.items() if key != "time"}
    return body


@pytest.mark.parametrize(("path", "params", "cached"), CASES)
async def test_both_prefixes_return_identical_payloads(versioned_client, path, params, cached):
    v1 = await versioned_client.get(f"{API_V1_PREFIX}{path}", params=params)
    legacy = await versioned_client.get(f"{API_LEGACY_PREFIX}{path}", params=params)

    assert v1.status_code == legacy.status_code == 200
    assert _comparable(path, v1.json()) == _comparable(path, legacy.json())
    if cached:
        # Same in-process cache behind both mounts: the second call is a hit and
        # serves the validator stored by the first.
        assert v1.headers["etag"] == legacy.headers["etag"]
        assert "max-age=" in legacy.headers["cache-control"]


async def test_health_stays_available_on_both_prefixes(versioned_client):
    """Render's healthCheckPath is ``/api/health``: it must not require a change."""
    for prefix in (API_LEGACY_PREFIX, API_V1_PREFIX):
        response = await versioned_client.get(f"{prefix}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


async def test_unknown_path_is_404_under_both_prefixes(versioned_client):
    for prefix in (API_LEGACY_PREFIX, API_V1_PREFIX):
        response = await versioned_client.get(f"{prefix}/nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


def test_every_route_is_mounted_under_both_prefixes(app):
    """No route may exist on one mount only (or twice under the same prefix)."""
    mounted = [(route, path) for route, path in _mounted_routes(app) if isinstance(route, APIRoute)]
    paths = {path for _, path in mounted}
    # One mount per (path, method) pair: nothing may be included twice.
    keys = {(path, frozenset(route.methods)) for route, path in mounted}
    assert len(keys) == len(mounted), "a route is mounted twice under the same prefix"
    legacy = {path for path in paths if path.startswith(f"{API_LEGACY_PREFIX}/")}
    v1 = {path for path in paths if path.startswith(f"{API_V1_PREFIX}/")}
    legacy -= {path for path in legacy if path.startswith(f"{API_V1_PREFIX}/")}

    assert legacy, "no legacy routes mounted"
    assert {path.replace(API_LEGACY_PREFIX, API_V1_PREFIX, 1) for path in legacy} == v1


def test_both_prefixes_share_the_same_endpoint_functions(app):
    """Proves the alias is a mount, not duplicated handler code."""
    endpoints = {
        path: route.endpoint for route, path in _mounted_routes(app) if isinstance(route, APIRoute)
    }

    for legacy_path, v1_path in PAIRS:
        assert endpoints[legacy_path] is endpoints[v1_path]


def test_openapi_marks_the_legacy_alias_deprecated(app):
    paths = app.openapi()["paths"]

    assert "/api/v1/health" in paths
    assert "/api/health" in paths
    assert paths["/api/health"]["get"]["deprecated"] is True
    assert not paths["/api/v1/health"]["get"].get("deprecated", False)
    # Distinct operation ids, otherwise the generated client/schema collides.
    assert (
        paths["/api/health"]["get"]["operationId"] != paths["/api/v1/health"]["get"]["operationId"]
    )
