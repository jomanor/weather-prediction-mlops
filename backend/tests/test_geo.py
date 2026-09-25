"""Open-Meteo geocoding: mapping, ordering, empty input, and upstream failures.

The service is exercised with ``httpx.MockTransport``; no test hits the network.
"""

import httpx
import pytest

from app.core.config import Settings
from app.services.geo import GeocodingError, GeocodingService, get_geo_service
from tests.fakes import FakeGeocodingService


def _service(handler: object) -> GeocodingService:
    return GeocodingService(
        Settings(_env_file=None, geo_timeout_seconds=8.0),
        transport=httpx.MockTransport(handler),
    )


async def test_search_maps_results_and_keeps_upstream_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["name"] == "Valencia"
        assert request.url.params["count"] == "8"
        assert request.url.params["language"] == "es"
        assert request.url.params["format"] == "json"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Valencia",
                        "latitude": 39.4699,
                        "longitude": -0.3763,
                        "country": "España",
                        "admin1": "Valencia",
                    },
                    {"name": "Valencia", "latitude": 10.0, "longitude": 20.0},
                ]
            },
        )

    service = _service(handler)
    try:
        results = await service.search(" Valencia ")
    finally:
        await service.aclose()

    assert [result.name for result in results] == ["Valencia", "Valencia"]
    assert results[0].country == "España"
    assert results[0].admin1 == "Valencia"
    assert results[1].country is None
    assert results[1].admin1 is None


async def test_search_empty_query_returns_empty_without_calling_upstream():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("upstream must not be called for an empty query")

    service = _service(handler)
    try:
        assert await service.search("   ") == []
    finally:
        await service.aclose()


@pytest.mark.parametrize("payload", [{"results": []}, {}])
async def test_search_empty_upstream_results_yields_empty(payload):
    service = _service(lambda request: httpx.Response(200, json=payload))
    try:
        assert await service.search("Atlantis") == []
    finally:
        await service.aclose()


async def test_search_skips_results_missing_name_or_coordinates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"latitude": 1.0, "longitude": 2.0},
                    {"name": "Valencia", "longitude": 2.0},
                    {"name": "Valencia", "latitude": 1.0, "longitude": 2.0},
                ]
            },
        )

    service = _service(handler)
    try:
        results = await service.search("Valencia")
    finally:
        await service.aclose()

    assert len(results) == 1


async def test_search_upstream_http_error_raises():
    service = _service(lambda request: httpx.Response(500, text="boom"))
    try:
        with pytest.raises(GeocodingError):
            await service.search("Valencia")
    finally:
        await service.aclose()


async def test_search_upstream_network_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    service = _service(handler)
    try:
        with pytest.raises(GeocodingError):
            await service.search("Valencia")
    finally:
        await service.aclose()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


async def test_geo_search_route_maps_results(client):
    response = await client.get("/api/geo/search", params={"q": "Valencia"})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "name": "Valencia",
                "latitude": 39.4699,
                "longitude": -0.3763,
                "country": "España",
                "admin1": "Valencia",
            }
        ]
    }


async def test_geo_search_route_without_query_is_empty(client):
    response = await client.get("/api/geo/search")

    assert response.status_code == 200
    assert response.json() == {"results": []}


async def test_geo_search_route_upstream_failure_is_502(client, app):
    app.dependency_overrides[get_geo_service] = lambda: FakeGeocodingService(
        error="Geocoding request failed: boom"
    )

    response = await client.get("/api/geo/search", params={"q": "Valencia"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Geocoding request failed: boom"}


async def test_geo_search_route_is_cached_per_query(client, geo):
    await client.get("/api/geo/search", params={"q": "Valencia"})
    await client.get("/api/geo/search", params={"q": " Valencia "})
    assert geo.queries == ["Valencia"]  # second call served from cache

    await client.get("/api/geo/search", params={"q": "Madrid"})
    assert geo.queries == ["Valencia", "Madrid"]  # distinct key
