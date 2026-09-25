"""Settings and CORS behaviour."""

from app.core.config import Settings


def test_defaults(monkeypatch):
    for key in ("MONGO_URI", "MONGO_URL", "CORS_ORIGINS", "AEMET_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.mongo_db == "weather_db"
    assert settings.cors_origins_list == ["http://localhost:5173"]
    assert settings.aemet_api_key is None
    assert settings.aemet_cache_ttl_seconds == 3600
    assert settings.ensure_indexes_on_start is True


def test_cors_origins_comma_separated(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test ,")
    settings = Settings(_env_file=None)
    assert settings.cors_origins_list == ["http://a.test", "http://b.test"]


def test_mongo_url_is_an_alias_for_mongo_uri(monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.setenv("MONGO_URL", "mongodb://legacy:27017")
    assert Settings(_env_file=None).mongo_uri == "mongodb://legacy:27017"


def test_mongo_uri_wins_over_alias(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://primary:27017")
    monkeypatch.setenv("MONGO_URL", "mongodb://legacy:27017")
    assert Settings(_env_file=None).mongo_uri == "mongodb://primary:27017"


async def test_cors_allows_configured_origin_and_rejects_others(client):
    allowed = await client.get("/api/health", headers={"Origin": "http://example.com"})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://example.com"

    denied = await client.get("/api/health", headers={"Origin": "http://evil.test"})
    assert denied.status_code == 200
    assert "access-control-allow-origin" not in denied.headers


async def test_cors_preflight(client):
    response = await client.options(
        "/api/weather/current",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://example.com"


async def test_cors_exposes_validator_and_freshness_headers(client):
    response = await client.get("/api/health", headers={"Origin": "http://example.com"})

    exposed = response.headers["access-control-expose-headers"].lower()
    for header in ("etag", "cache-control", "x-data-age-seconds"):
        assert header in exposed


async def test_request_id_header_is_echoed(client):
    response = await client.get("/api/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"
