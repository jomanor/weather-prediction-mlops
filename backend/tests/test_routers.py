"""Router happy paths, 404s and contract-shaped payloads (offline, fake repos)."""

from datetime import timedelta

from app.core.thresholds import PREDICTION_STALE_AFTER_HOURS
from app.repositories.prediction_repo import get_prediction_repo
from app.repositories.weather_repo import WeatherRepository, get_weather_repo
from app.schemas.predictions import Prediction, freshness_fields
from app.schemas.quality import CityQuality
from app.schemas.weather import CurrentWeather
from app.services.aemet import get_aemet_service
from tests.fakes import (
    FakeAemetService,
    FakeDatabase,
    FakeMongoCollection,
    FakeMongoDb,
    FakePredictionRepository,
    FakeWeatherRepository,
)


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
        "is_day",
        "observed_at",
    }
    assert body["is_day"] == 1
    assert body["observed_at"].endswith("Z")


async def test_current_weather_unknown_city_is_404(client):
    response = await client.get("/api/weather/current/Atlantis")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["message"] == "City 'Atlantis' not found"


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
        "temp_lower",
        "temp_upper",
        "interval_level",
        "temp_model_name",
        "temp_model_version",
        "rain_model_name",
        "rain_model_version",
        "source_age_hours",
        "stale",
    }
    assert prediction["city"] == "Madrid"
    # The interval is passed through verbatim; the interval-less legacy doc would
    # serialize these as null instead (covered in the repository mapping tests).
    assert (prediction["temp_lower"], prediction["temp_upper"]) == (19.6, 23.4)
    assert prediction["interval_level"] == 0.8


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
    assert "Atlantis" in response.json()["error"]["message"]


async def test_predictions_latest_returns_every_horizon(client, app, multi_horizon_predictions):
    app.dependency_overrides[get_prediction_repo] = lambda: FakePredictionRepository(
        multi_horizon_predictions
    )

    response = await client.get("/api/predictions/latest")

    assert response.status_code == 200
    body = response.json()
    # Omitted horizon -> one row per (city, horizon_hours).
    assert body["count"] == 2
    assert [p["horizon_hours"] for p in body["predictions"]] == [1, 3]
    # The older h1 row lost to the fresh one.
    assert [p["predicted_temperature"] for p in body["predictions"]] == [21.5, 22.0]
    assert body["generated_at"] == body["predictions"][0]["prediction_timestamp"]


async def test_predictions_latest_filters_one_horizon(client, app, multi_horizon_predictions):
    app.dependency_overrides[get_prediction_repo] = lambda: FakePredictionRepository(
        multi_horizon_predictions
    )

    response = await client.get("/api/predictions/latest", params={"horizon": 1})

    assert response.status_code == 200
    body = response.json()
    # Provided horizon -> one row per city.
    assert body["count"] == 1
    assert body["predictions"][0]["horizon_hours"] == 1
    assert body["predictions"][0]["predicted_temperature"] == 21.5


async def test_predictions_latest_rejects_out_of_range_horizon(client):
    for horizon in (0, 49):
        response = await client.get("/api/predictions/latest", params={"horizon": horizon})
        assert response.status_code == 422, horizon


async def test_predictions_for_city_filters_horizon(client, app, multi_horizon_predictions):
    app.dependency_overrides[get_prediction_repo] = lambda: FakePredictionRepository(
        multi_horizon_predictions
    )

    response = await client.get("/api/predictions/Madrid", params={"horizon": 3})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["horizon_hours"] == 3
    assert body[0]["predicted_temperature"] == 22.0


async def test_predictions_latest_fresh_row_is_not_stale(client):
    response = await client.get("/api/predictions/latest")

    assert response.status_code == 200
    prediction = response.json()["predictions"][0]
    # ``prediction_points`` is 1 h old; the wall clock has advanced a little by
    # the time the route serializes it, so allow a small tolerance.
    assert prediction["stale"] is False
    assert abs(prediction["source_age_hours"] - 1.0) < 0.2


async def test_predictions_latest_flags_a_35_day_old_row_without_dropping_it(client, app, now):
    stale_row = Prediction(
        city="Sevilla",
        source_timestamp=now - timedelta(days=35),
        prediction_timestamp=now - timedelta(days=35) + timedelta(hours=1),
        horizon_hours=1,
        predicted_temperature=18.0,
    )
    fresh_row = Prediction(
        city="Madrid",
        source_timestamp=now - timedelta(hours=1),
        prediction_timestamp=now,
        horizon_hours=1,
        predicted_temperature=21.0,
    )
    app.dependency_overrides[get_prediction_repo] = lambda: FakePredictionRepository(
        [stale_row, fresh_row]
    )

    response = await client.get("/api/predictions/latest", params={"horizon": 1})

    assert response.status_code == 200
    rows = {p["city"]: p for p in response.json()["predictions"]}
    # A stale row stays visible and flagged, never dropped.
    assert set(rows) == {"Madrid", "Sevilla"}
    assert rows["Sevilla"]["stale"] is True
    assert abs(rows["Sevilla"]["source_age_hours"] - 35 * 24) < 0.2
    assert rows["Madrid"]["stale"] is False
    assert abs(rows["Madrid"]["source_age_hours"] - 1.0) < 0.2


async def test_predictions_freshness_fields_survive_both_prefixes(client):
    for prefix in ("/api", "/api/v1"):
        latest = await client.get(f"{prefix}/predictions/latest")
        assert latest.status_code == 200, prefix
        assert "source_age_hours" in latest.json()["predictions"][0], prefix
        assert "stale" in latest.json()["predictions"][0], prefix

        city = await client.get(f"{prefix}/predictions/Madrid")
        assert city.status_code == 200, prefix
        assert all("source_age_hours" in row for row in city.json()), prefix
        assert all("stale" in row for row in city.json()), prefix


def test_freshness_fields_never_invent_an_age(now):
    threshold = PREDICTION_STALE_AFTER_HOURS
    assert freshness_fields(None, now, threshold) == {"source_age_hours": None, "stale": False}
    # Exactly at the threshold is not stale; past it is.
    assert freshness_fields(now - timedelta(hours=threshold), now, threshold) == {
        "source_age_hours": threshold,
        "stale": False,
    }
    assert freshness_fields(now - timedelta(hours=threshold + 0.1), now, threshold)["stale"] is True
    assert freshness_fields(now - timedelta(hours=1), now, threshold)["stale"] is False


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
    first = body["models"][0]
    assert first["name"] == "temp_prediction_1h_GradientBoostedTrees"
    assert first["version"] == "20260923_020000"
    assert first["target"] == "temperature"
    assert first["horizon_hours"] == 1
    assert first["stage"] == "production"
    # Honest-metrics block (Batch 3): present keys carry the registry values,
    # absent ones stay null instead of a fabricated 0.
    assert first["metrics"] == {
        "rmse": 1.44,
        "mae": 1.12,
        "r2": 0.91,
        "persistence_rmse": 3.05,
        "climatology_rmse": 3.41,
        "skill_score": 0.53,
        "brier": None,
        "persistence_brier": None,
        "prevalence": None,
        "coverage": 0.79,
    }
    assert first["split"] == {
        "kind": "temporal",
        "train_end": "2026-09-16T00:00:00Z",
        "val_end": "2026-09-19T00:00:00Z",
        "test_start": "2026-09-19T00:00:00Z",
    }
    assert first["interval"] == {"level": 0.8, "lower_offset": -1.9, "upper_offset": 2.1}
    assert first["commit"] == "abc1234"
    # Batch 4, Contract 3: the diagnostics block is surfaced additively.
    diagnostics = first["diagnostics"]
    assert diagnostics["by_city"] == [
        {"label": "Madrid", "n": 24, "rmse": 1.2, "mae": 1.0, "bias": -0.1, "brier": None},
        {"label": "Alicante", "n": 0, "rmse": None, "mae": None, "bias": None, "brier": None},
    ]
    assert diagnostics["by_hour_of_day"] == [
        {"label": "0", "n": 10, "rmse": 1.3, "mae": 1.1, "bias": 0.0, "brier": None}
    ]
    assert diagnostics["by_rain_bucket"][0]["brier"] == 0.12
    assert diagnostics["drift_psi"] == {"temperature": 0.08, "humidity": 0.31, "pressure": None}
    # fields the registry document does not carry stay null, never invented
    assert body["models"][1]["metrics"] is None
    assert body["models"][1]["stage"] is None
    assert body["models"][1]["split"] is None
    assert body["models"][1]["interval"] is None
    assert body["models"][1]["commit"] is None
    # A legacy registry doc has no diagnostics; it must stay None, not {}.
    assert body["models"][1]["diagnostics"] is None


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
        "/api/weather/series",
        "/api/weather/quality",
        "/api/map/stations",
        "/api/predictions/latest",
        "/api/predictions/{city}",
        "/api/benchmark",
        "/api/benchmark/{city}",
        "/api/models",
    } <= set(schema["paths"])
    # Finding 13: the zero-caller reads were deleted, not left mounted.
    assert not {
        "/api/weather/stats/{city}",
        "/api/weather/summary",
        "/api/weather/range/{city}",
    } & set(schema["paths"])


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


async def test_if_none_match_accepts_a_comma_separated_list(client):
    first = await client.get("/api/cities")
    etag = first.headers["etag"]

    matched = await client.get("/api/cities", headers={"If-None-Match": f'"bogus",  {etag}'})
    assert matched.status_code == 304

    unmatched = await client.get("/api/cities", headers={"If-None-Match": '"bogus", "other"'})
    assert unmatched.status_code == 200


async def test_if_none_match_star_matches_any_representation(client):
    await client.get("/api/cities")
    response = await client.get("/api/cities", headers={"If-None-Match": "*"})
    assert response.status_code == 304


async def test_if_none_match_weak_validator_matches(client):
    first = await client.get("/api/cities")
    response = await client.get(
        "/api/cities", headers={"If-None-Match": f"W/{first.headers['etag']}"}
    )
    assert response.status_code == 304


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
# L4 - bulk series and the station map
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


async def test_series_rejects_overlong_city_name(client):
    response = await client.get("/api/weather/series", params={"cities": "M" * 101})
    assert response.status_code == 422


async def test_series_rejects_out_of_range_step_hours(client):
    for step in (0, 25):
        response = await client.get(
            "/api/weather/series", params={"cities": "Madrid", "step_hours": step}
        )
        assert response.status_code == 422, step


async def test_series_step_hours_downsamples(client, app, now):
    points = [
        CurrentWeather(
            city="Madrid",
            temperature=float(index),
            observed_at=now - timedelta(hours=index),
        )
        for index in range(7)
    ]
    repo = FakeWeatherRepository(points)
    app.dependency_overrides[get_weather_repo] = lambda: repo

    # Seven consecutive hourly points span exactly two 6 h buckets.
    sampled = await client.get(
        "/api/weather/series",
        params={"cities": "Madrid", "hours": 24, "step_hours": 6},
    )
    assert sampled.status_code == 200
    assert repo.last_step_hours == 6
    assert sampled.json()["cities"][0]["count"] == 2

    unsampled = await client.get(
        "/api/weather/series",
        params={"cities": "Madrid", "hours": 24, "step_hours": 1},
    )
    assert unsampled.json()["cities"][0]["count"] == 7


async def test_series_raw_fallback_returns_coordinates(client, app, now):
    raw = FakeMongoCollection(
        [
            {
                "city": "Alicante",
                "timestamp": now - timedelta(hours=1),
                "payload": {
                    "latitude": 38.3452,
                    "longitude": -0.481,
                    "current": {"temperature_2m": 25.0, "is_day": 0},
                },
            }
        ]
    )
    repo = WeatherRepository(FakeMongoDb(FakeMongoCollection([]), raw))
    app.dependency_overrides[get_weather_repo] = lambda: repo

    response = await client.get("/api/weather/series", params={"cities": "Alicante", "hours": 24})

    assert response.status_code == 200
    point = response.json()["cities"][0]["points"][0]
    assert point["latitude"] == 38.3452
    assert point["longitude"] == -0.481
    assert point["temperature"] == 25.0
    # ``is_day`` is part of the public payload and survives the projection used
    # when ``fields`` is passed.
    assert point["is_day"] == 0

    projected = await client.get(
        "/api/weather/series",
        params={"cities": "Alicante", "hours": 24, "fields": "is_day"},
    )
    assert projected.status_code == 200
    assert projected.json()["cities"][0]["points"][0]["is_day"] == 0


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
    # The map glyph set needs the station's own day/night flag, not the viewer's clock.
    assert properties["is_day"] == 1
    assert properties["observed_at"].endswith("Z")

    second = await client.get("/api/map/stations", headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 304
    assert weather_repo.calls.count("latest_per_city") == 1


async def test_quality_returns_meter_shape_and_caches(client, app, now):
    quality = [
        CityQuality(
            city="Madrid",
            expected_hours=168,
            observed_hours=165,
            completeness=165 / 168,
            max_gap_hours=3.0,
            null_rate=0.012,
            last_observed_at=now - timedelta(hours=1),
            age_hours=1.0,
            status="ok",
        )
    ]
    repo = FakeWeatherRepository(quality=quality)
    app.dependency_overrides[get_weather_repo] = lambda: repo

    first = await client.get("/api/weather/quality", params={"days": 7})

    assert first.status_code == 200
    body = first.json()
    assert body["days"] == 7
    assert body["generated_at"].endswith("Z")
    assert len(body["cities"]) == 1
    city = body["cities"][0]
    assert set(city) == {
        "city",
        "expected_hours",
        "observed_hours",
        "completeness",
        "max_gap_hours",
        "null_rate",
        "last_observed_at",
        "age_hours",
        "status",
    }
    assert city["city"] == "Madrid"
    assert city["expected_hours"] == 168
    assert city["observed_hours"] == 165
    assert city["max_gap_hours"] == 3.0
    assert city["null_rate"] == 0.012
    assert city["age_hours"] == 1.0
    assert city["status"] == "ok"
    assert city["last_observed_at"].endswith("Z")
    assert "max-age=300" in first.headers["cache-control"]

    second = await client.get(
        "/api/weather/quality", params={"days": 7}, headers={"If-None-Match": first.headers["etag"]}
    )
    assert second.status_code == 304
    assert repo.calls.count("quality_report") == 1
    assert repo.last_quality_window[0] == 7


async def test_quality_passes_days_to_the_repo(client, app, weather_repo):
    response = await client.get("/api/weather/quality", params={"days": 30})

    assert response.status_code == 200
    assert weather_repo.last_quality_window[0] == 30


async def test_quality_rejects_out_of_range_days(client):
    for days in (0, 31):
        response = await client.get("/api/weather/quality", params={"days": days})
        assert response.status_code == 422, days


async def test_quality_accepts_boundary_days(client, weather_repo):
    for days in (1, 30):
        response = await client.get("/api/weather/quality", params={"days": days})
        assert response.status_code == 200, days
        assert weather_repo.last_quality_window[0] == days


# ---------------------------------------------------------------------------
# Finding 11(b): the two heaviest reads go through the shared TTL cache, and the
# key covers every query parameter that changes the payload.
# ---------------------------------------------------------------------------


async def test_predictions_latest_is_cached_for_30s(client, prediction_repo):
    first = await client.get("/api/predictions/latest")
    second = await client.get("/api/predictions/latest")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert "max-age=30" in first.headers["cache-control"]
    assert first.headers["etag"] == second.headers["etag"]
    # One aggregate for two requests.
    assert prediction_repo.calls.count("latest_per_city") == 1


async def test_predictions_latest_304_is_served_from_the_cached_payload(client, prediction_repo):
    first = await client.get("/api/predictions/latest")

    second = await client.get(
        "/api/predictions/latest", headers={"If-None-Match": first.headers["etag"]}
    )

    assert second.status_code == 304
    assert second.headers["etag"] == first.headers["etag"]
    assert prediction_repo.calls.count("latest_per_city") == 1


async def test_predictions_latest_cache_key_includes_the_horizon(
    client, app, multi_horizon_predictions
):
    repository = FakePredictionRepository(multi_horizon_predictions)
    app.dependency_overrides[get_prediction_repo] = lambda: repository

    every_horizon = await client.get("/api/predictions/latest")
    one_horizon = await client.get("/api/predictions/latest", params={"horizon": 1})
    repeat = await client.get("/api/predictions/latest", params={"horizon": 1})

    # Different horizon -> different key -> different payload.
    assert every_horizon.json()["count"] == 2
    assert one_horizon.json()["count"] == 1
    assert repeat.json() == one_horizon.json()
    assert repository.calls.count("latest_per_city") == 2


async def test_weather_series_is_cached_for_30s(client, weather_repo):
    first = await client.get("/api/weather/series", params={"cities": "Madrid", "hours": 24})
    second = await client.get("/api/weather/series", params={"cities": "Madrid", "hours": 24})

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert "max-age=30" in first.headers["cache-control"]
    assert weather_repo.calls.count("find_many_in_range") == 1


async def test_weather_series_cache_key_covers_every_payload_parameter(client, weather_repo):
    base = {"cities": "Madrid", "hours": 24, "fields": "temperature", "step_hours": 1}
    await client.get("/api/weather/series", params=base)

    # A repeat of the same request is a hit.
    await client.get("/api/weather/series", params=base)
    assert weather_repo.calls.count("find_many_in_range") == 1

    # Each differing parameter is a miss: hours, step_hours, fields, cities.
    for override in (
        {"hours": 48},
        {"step_hours": 3},
        {"fields": "temperature,humidity"},
        {"cities": "Madrid,Alicante"},
    ):
        await client.get("/api/weather/series", params={**base, **override})
    assert weather_repo.calls.count("find_many_in_range") == 5


async def test_weather_series_key_normalises_the_city_list(client, weather_repo):
    await client.get("/api/weather/series", params={"cities": "Madrid", "fields": "temperature"})

    # Same payload, different spelling: blanks and a duplicate are normalised
    # away by ``_parse_cities`` and must hit the same entry.
    response = await client.get(
        "/api/weather/series", params={"cities": " Madrid , Madrid", "fields": "temperature"}
    )

    assert response.status_code == 200
    assert weather_repo.calls.count("find_many_in_range") == 1


async def test_weather_series_key_treats_the_field_list_as_a_set(client, weather_repo):
    await client.get(
        "/api/weather/series", params={"cities": "Madrid", "fields": "temperature,humidity"}
    )

    # A reordered projection produces the same payload, so it must not miss.
    response = await client.get(
        "/api/weather/series", params={"cities": "Madrid", "fields": "humidity,temperature"}
    )

    assert response.status_code == 200
    assert weather_repo.calls.count("find_many_in_range") == 1


# ---------------------------------------------------------------------------
# Finding 5: the freshness badge reads the one shared threshold, never 24 h
# ---------------------------------------------------------------------------


async def test_the_stale_badge_uses_the_shared_12_hour_threshold(client, app, now):
    def row(city: str, age_hours: float) -> Prediction:
        return Prediction(
            city=city,
            source_timestamp=now - timedelta(hours=age_hours),
            prediction_timestamp=now,
            horizon_hours=1,
            predicted_temperature=20.0,
        )

    app.dependency_overrides[get_prediction_repo] = lambda: FakePredictionRepository(
        [row("Fresh", 11.0), row("Old", 13.0)]
    )

    body = (await client.get("/api/predictions/latest")).json()
    rows = {prediction["city"]: prediction for prediction in body["predictions"]}

    # 13 h is stale: the old 24 h setting would have called it fresh.
    assert PREDICTION_STALE_AFTER_HOURS == 12.0
    assert rows["Fresh"]["stale"] is False
    assert rows["Old"]["stale"] is True
