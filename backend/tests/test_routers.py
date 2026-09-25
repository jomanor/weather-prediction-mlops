"""Router happy paths, 404s and contract-shaped payloads (offline, fake repos)."""

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
        "/api/predictions/latest",
        "/api/predictions/{city}",
        "/api/benchmark",
        "/api/benchmark/{city}",
        "/api/models",
    } <= set(schema["paths"])
