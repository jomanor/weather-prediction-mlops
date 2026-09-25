"""Router happy paths, 404s and contract-shaped payloads (offline, fake repos)."""

from datetime import timedelta

from app.services.aemet import get_aemet_service
from tests.fakes import FakeAemetService, FakeDatabase


async def test_health(client):
    response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "weather-api"
    assert body["version"] == "2.0.0"
    assert body["time"].endswith("Z")


async def test_health_liveness_is_shallow(client):
    # No ``app.state.db`` is set: liveness must not touch Mongo.
    response = await client.get("/api/health")
    assert response.status_code == 200


async def test_health_ready_ok(client, app):
    app.state.db = FakeDatabase()
    response = await client.get("/api/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "weather_db"}


async def test_health_ready_unavailable_when_mongo_is_down(client, app):
    app.state.db = FakeDatabase(error=RuntimeError("no mongod"))
    response = await client.get("/api/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "weather_db"}


async def test_cities(client):
    response = await client.get("/api/cities")
    assert response.status_code == 200
    assert response.json() == [
        {"name": "Alicante", "latitude": 38.3452, "longitude": -0.481},
        {"name": "Madrid", "latitude": 40.4168, "longitude": -3.7038},
    ]


async def test_current_weather_all(client):
    response = await client.get("/api/weather/current")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert len(body["stations"]) == 1
    assert body["stations"][0]["temperature"] == 22.0


async def test_current_weather_for_city_has_every_contract_field(client):
    response = await client.get("/api/weather/current/Madrid")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "city",
        "latitude",
        "longitude",
        "temperature",
        "apparent_temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "precipitation",
        "cloud_cover",
        "weather_code",
        "observed_at",
    }
    assert body["observed_at"].endswith("Z")


async def test_current_weather_unknown_city_is_404(client):
    response = await client.get("/api/weather/current/Atlantis")
    assert response.status_code == 404
    assert response.json() == {"detail": "City 'Atlantis' not found"}


async def test_history_is_chronological(client):
    response = await client.get("/api/weather/history/Madrid", params={"hours": 24, "limit": 200})

    assert response.status_code == 200
    body = response.json()
    assert body["city"] == "Madrid"
    assert body["hours"] == 24
    assert body["count"] == 3
    temperatures = [point["temperature"] for point in body["points"]]
    # Charts read left to right from oldest to newest.
    assert temperatures == [20.0, 21.0, 22.0]


async def test_history_without_data_is_404(client):
    response = await client.get("/api/weather/history/Atlantis")
    assert response.status_code == 404


async def test_history_rejects_out_of_range_hours(client):
    response = await client.get("/api/weather/history/Madrid", params={"hours": 0})
    assert response.status_code == 422


async def test_stats(client):
    response = await client.get("/api/weather/stats/Madrid", params={"hours": 24})

    assert response.status_code == 200
    body = response.json()
    assert body["city"] == "Madrid"
    assert body["count"] == 3
    assert body["temperature"] == {"avg": 21.0, "min": 20.0, "max": 22.0}
    assert body["humidity"] == {"avg": 60.0, "min": 60.0, "max": 60.0}
    assert body["pressure"]["avg"] == 1014.0
    assert body["wind_speed"]["avg"] == 10.0
    assert body["precipitation_total"] == 0.0
    assert body["start"].endswith("Z") and body["end"].endswith("Z")


async def test_stats_without_data_is_404(client):
    response = await client.get("/api/weather/stats/Atlantis")
    assert response.status_code == 404


async def test_predictions_latest(client):
    response = await client.get("/api/predictions/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["generated_at"].endswith("Z")
    prediction = body["predictions"][0]
    assert set(prediction) == {
        "city",
        "source_timestamp",
        "prediction_timestamp",
        "horizon_hours",
        "predicted_temperature",
        "predicted_rain",
        "observed_temperature",
        "temp_model_name",
        "temp_model_version",
        "rain_model_name",
        "rain_model_version",
    }
    assert prediction["city"] == "Madrid"


async def test_predictions_for_city_chronological(client):
    response = await client.get("/api/predictions/Madrid", params={"limit": 48})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    # Served oldest to newest so time-series charts read left to right.
    assert [item["predicted_temperature"] for item in body] == [20.4, 21.5]


async def test_predictions_for_unknown_city_is_404(client):
    response = await client.get("/api/predictions/Atlantis")
    assert response.status_code == 404
    assert "Atlantis" in response.json()["detail"]


async def test_benchmark_for_city(client):
    response = await client.get("/api/benchmark/Madrid", params={"hours": 24})

    assert response.status_code == 200
    body = response.json()
    assert body["city"] == "Madrid"
    assert body["hours"] == 24
    assert body["aemet"] == {
        "available": True,
        "error": None,
        "issued_at": body["aemet"]["issued_at"],
    }
    assert body["aemet"]["issued_at"].endswith("Z")
    assert len(body["series"]) == 3
    assert set(body["series"][0]) == {
        "timestamp",
        "observed",
        "model",
        "aemet",
        "residual_model",
        "residual_aemet",
    }
    assert body["metrics"]["model"] == {"mae": 0.45, "rmse": 0.45, "bias": 0.45, "n": 2}
    assert body["metrics"]["aemet"] == {"mae": 0.5, "rmse": 0.58, "bias": 0.5, "n": 2}


async def test_benchmark_for_unknown_city_is_404(client):
    response = await client.get("/api/benchmark/Atlantis")
    assert response.status_code == 404


async def test_benchmark_summary(client):
    response = await client.get("/api/benchmark")

    assert response.status_code == 200
    body = response.json()
    assert body["aemet_configured"] is True
    assert body["generated_at"].endswith("Z")
    assert len(body["cities"]) == 1
    assert body["cities"][0]["city"] == "Madrid"
    assert body["cities"][0]["n"] == 3
    assert body["cities"][0]["model"]["mae"] == 0.45


async def test_models(client):
    response = await client.get("/api/models")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["models"][0] == {
        "name": "temp_prediction_1h_GradientBoostedTrees",
        "version": "20260923_020000",
        "target": "temperature",
        "horizon_hours": 1,
        "created_at": body["models"][0]["created_at"],
        "metrics": {"rmse": 1.44, "mae": 1.12, "r2": 0.91},
        "stage": "production",
    }
    # fields the registry document does not carry stay null, never invented
    assert body["models"][1]["metrics"] is None
    assert body["models"][1]["stage"] is None


async def test_benchmark_without_aemet_key_still_returns_series(client, app):
    app.dependency_overrides[get_aemet_service] = lambda: FakeAemetService(
        configured=False, error="AEMET_API_KEY is not configured"
    )

    response = await client.get("/api/benchmark/Madrid")

    assert response.status_code == 200
    body = response.json()
    assert body["aemet"]["available"] is False
    assert body["aemet"]["error"] == "AEMET_API_KEY is not configured"
    assert len(body["series"]) == 3
    assert body["metrics"]["aemet"]["n"] == 0
    assert body["metrics"]["model"]["n"] == 2


async def test_openapi_documents_every_contract_path(client):
    schema = (await client.get("/openapi.json")).json()
    assert {
        "/api/health",
        "/api/health/ready",
        "/api/cities",
        "/api/geo/search",
        "/api/weather/current",
        "/api/weather/current/{city}",
        "/api/weather/history/{city}",
        "/api/weather/stats/{city}",
        "/api/weather/series",
        "/api/weather/summary",
        "/api/weather/range/{city}",
        "/api/map/stations",
        "/api/predictions/latest",
        "/api/predictions/{city}",
        "/api/benchmark",
        "/api/benchmark/{city}",
        "/api/models",
    } <= set(schema["paths"])


# ---------------------------------------------------------------------------
# L2 - TTL cache + HTTP validators
# ---------------------------------------------------------------------------


async def test_cities_cache_hit_skips_repository(client, city_repo):
    first = await client.get("/api/cities")
    second = await client.get("/api/cities")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert city_repo.calls.count("list") == 1


async def test_cities_etag_then_304_round_trip(client, city_repo):
    first = await client.get("/api/cities")
    etag = first.headers["etag"]
    assert "max-age=60" in first.headers["cache-control"]
    assert "stale-while-revalidate" in first.headers["cache-control"]

    second = await client.get("/api/cities", headers={"If-None-Match": etag})

    assert second.status_code == 304
    assert second.headers["etag"] == etag
    # The validator came from the cached payload, never from a second query.
    assert city_repo.calls.count("list") == 1


async def test_current_weather_cache_hit_and_data_age_header(client, weather_repo):
    first = await client.get("/api/weather/current")
    assert first.status_code == 200
    assert int(first.headers["x-data-age-seconds"]) >= 0

    second = await client.get(
        "/api/weather/current", headers={"If-None-Match": first.headers["etag"]}
    )

    assert second.status_code == 304
    assert weather_repo.calls.count("latest_per_city") == 1


async def test_current_weather_for_city_data_age_header(client, weather_repo):
    first = await client.get("/api/weather/current/Madrid")
    assert int(first.headers["x-data-age-seconds"]) >= 0

    second = await client.get(
        "/api/weather/current/Madrid", headers={"If-None-Match": first.headers["etag"]}
    )

    assert second.status_code == 304
    assert weather_repo.calls.count("latest_for_city") == 1


# ---------------------------------------------------------------------------
# L4 / L5 - bulk series, summary, map and range
# ---------------------------------------------------------------------------


async def test_series_bulk_returns_chronological_series(client):
    response = await client.get("/api/weather/series", params={"cities": "Madrid", "hours": 24})

    assert response.status_code == 200
    body = response.json()
    assert body["hours"] == 24
    assert body["count"] == 1
    series = body["cities"][0]
    assert series["city"] == "Madrid"
    assert series["count"] == 3
    assert series["latest_timestamp"].endswith("Z")
    assert [point["temperature"] for point in series["points"]] == [20.0, 21.0, 22.0]


async def test_series_rejects_unknown_field(client):
    response = await client.get(
        "/api/weather/series", params={"cities": "Madrid", "fields": "temperature,bogus"}
    )
    assert response.status_code == 422


async def test_series_rejects_too_many_cities(client):
    cities = ",".join(f"City{index}" for index in range(16))
    response = await client.get("/api/weather/series", params={"cities": cities})
    assert response.status_code == 422


async def test_series_rejects_empty_cities(client):
    response = await client.get("/api/weather/series", params={"cities": " , "})
    assert response.status_code == 422


async def test_series_rejects_out_of_range_hours(client):
    response = await client.get("/api/weather/series", params={"cities": "Madrid", "hours": 169})
    assert response.status_code == 422


async def test_summary_computes_aggregates_and_sparkline(client):
    response = await client.get("/api/weather/summary", params={"cities": "Madrid", "hours": 24})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    summary = body["cities"][0]
    assert summary["city"] == "Madrid"
    assert summary["min"] == 20.0
    assert summary["max"] == 22.0
    assert summary["first"] == 20.0
    assert summary["last"] == 22.0
    assert summary["trend"] == 2.0
    assert summary["points"] == [20.0, 21.0, 22.0]


async def test_summary_rejects_too_many_cities(client):
    cities = ",".join(f"City{index}" for index in range(16))
    response = await client.get("/api/weather/summary", params={"cities": cities})
    assert response.status_code == 422


async def test_map_stations_returns_geojson_and_caches(client, weather_repo):
    first = await client.get("/api/map/stations")

    assert first.status_code == 200
    body = first.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 1
    feature = body["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"] == {"type": "Point", "coordinates": [-3.7038, 40.4168]}
    properties = feature["properties"]
    assert properties["city"] == "Madrid"
    assert properties["temperature"] == 22.0
    assert properties["relative_humidity"] == 60.0
    assert properties["observed_at"].endswith("Z")

    second = await client.get("/api/map/stations", headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 304
    assert weather_repo.calls.count("latest_per_city") == 1


async def test_range_paginates_with_cursor_oldest_first(client, now):
    start = (now - timedelta(hours=3)).isoformat()
    end = (now + timedelta(hours=1)).isoformat()

    first = await client.get(
        "/api/weather/range/Madrid", params={"from": start, "to": end, "limit": 2}
    )

    assert first.status_code == 200
    body = first.json()
    assert [point["temperature"] for point in body["points"]] == [20.0, 21.0]
    assert body["next_cursor"] is not None

    second = await client.get(
        "/api/weather/range/Madrid",
        params={"from": start, "to": end, "limit": 2, "cursor": body["next_cursor"]},
    )

    assert second.status_code == 200
    body = second.json()
    assert [point["temperature"] for point in body["points"]] == [22.0]
    assert body["next_cursor"] is None


async def test_range_rejects_inverted_window(client, now):
    response = await client.get(
        "/api/weather/range/Madrid",
        params={"from": now.isoformat(), "to": (now - timedelta(hours=1)).isoformat()},
    )
    assert response.status_code == 422


async def test_range_rejects_malformed_timestamp(client):
    response = await client.get(
        "/api/weather/range/Madrid", params={"from": "not-a-date", "to": "also-bad"}
    )
    assert response.status_code == 422


async def test_range_caps_limit(client, now):
    response = await client.get(
        "/api/weather/range/Madrid",
        params={
            "from": (now - timedelta(hours=3)).isoformat(),
            "to": (now + timedelta(hours=1)).isoformat(),
            "limit": 2001,
        },
    )
    assert response.status_code == 422
