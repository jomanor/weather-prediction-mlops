"""Shared fixtures: an app with fake dependencies and an offline HTTP client."""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app
from app.repositories.city_repo import get_city_repo
from app.repositories.model_repo import get_model_repo
from app.repositories.prediction_repo import get_prediction_repo
from app.repositories.weather_repo import get_weather_repo
from app.schemas.city import City, GeoResult
from app.schemas.models import ModelInfo, ModelMetrics
from app.schemas.predictions import Prediction
from app.schemas.weather import CurrentWeather
from app.services.aemet import get_aemet_service
from app.services.geo import get_geo_service
from tests.fakes import (
    FakeAemetService,
    FakeCityRepository,
    FakeGeocodingService,
    FakeModelRepository,
    FakePredictionRepository,
    FakeWeatherRepository,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        mongo_uri="mongodb://localhost:27017",
        cors_origins="http://localhost:5173,http://example.com",
    )


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@pytest.fixture
def weather_points(now: datetime) -> list[CurrentWeather]:
    """Three Madrid observations: now (22.0), now-1h (21.0), now-2h (20.0)."""
    return [
        CurrentWeather(
            city="Madrid",
            latitude=40.4168,
            longitude=-3.7038,
            temperature=temperature,
            apparent_temperature=temperature - 0.5,
            humidity=60.0,
            pressure=1014.0,
            wind_speed=10.0,
            wind_direction=200.0,
            precipitation=0.0,
            cloud_cover=30.0,
            weather_code=2,
            observed_at=now - timedelta(hours=offset),
        )
        for offset, temperature in ((0, 22.0), (1, 21.0), (2, 20.0))
    ]


@pytest.fixture
def prediction_points(now: datetime) -> list[Prediction]:
    """Two model predictions for the two older observation timestamps."""
    return [
        Prediction(
            city="Madrid",
            source_timestamp=now - timedelta(hours=offset),
            prediction_timestamp=now,
            horizon_hours=1,
            predicted_temperature=predicted,
            predicted_rain=0.0,
            observed_temperature=observed,
            temp_model_name="temp_prediction_1h_GradientBoostedTrees",
            temp_model_version="20260923_020000",
            rain_model_name="rain_prediction_1h_GradientBoostedTrees",
            rain_model_version="20260923_020000",
        )
        for offset, predicted, observed in ((1, 21.5, 21.0), (2, 20.4, 20.0))
    ]


@pytest.fixture
def aemet_temps(now: datetime) -> dict[datetime, float]:
    return {now - timedelta(hours=1): 21.8, now - timedelta(hours=2): 20.2}


@pytest.fixture
def model_infos(now: datetime) -> list[ModelInfo]:
    return [
        ModelInfo(
            name="temp_prediction_1h_GradientBoostedTrees",
            version="20260923_020000",
            target="temperature",
            horizon_hours=1,
            created_at=now - timedelta(hours=12),
            metrics=ModelMetrics(rmse=1.44, mae=1.12, r2=0.91),
            stage="production",
        ),
        ModelInfo(
            name="rain_prediction_1h_RandomForest",
            version="20260923_020000",
            target="rain",
            horizon_hours=1,
            created_at=now - timedelta(hours=12),
            metrics=None,
            stage=None,
        ),
    ]


@pytest.fixture
def weather_repo(weather_points) -> FakeWeatherRepository:
    return FakeWeatherRepository(weather_points)


@pytest.fixture
def prediction_repo(prediction_points) -> FakePredictionRepository:
    return FakePredictionRepository(prediction_points)


@pytest.fixture
def model_repo(model_infos) -> FakeModelRepository:
    return FakeModelRepository(model_infos)


@pytest.fixture
def aemet(aemet_temps, now) -> FakeAemetService:
    return FakeAemetService(temps=aemet_temps, issued_at=now - timedelta(hours=8))


@pytest.fixture
def city_repo() -> FakeCityRepository:
    return FakeCityRepository(
        [
            City(name="Madrid", latitude=40.4168, longitude=-3.7038),
            City(name="Alicante", latitude=38.3452, longitude=-0.4810),
        ]
    )


@pytest.fixture
def geo() -> FakeGeocodingService:
    return FakeGeocodingService(
        results=[
            GeoResult(
                name="Valencia",
                latitude=39.4699,
                longitude=-0.3763,
                country="España",
                admin1="Valencia",
            )
        ]
    )


@pytest.fixture
def app(
    settings,
    weather_repo,
    prediction_repo,
    model_repo,
    aemet,
    city_repo,
    geo,
):
    application = create_app(settings)
    application.dependency_overrides[get_weather_repo] = lambda: weather_repo
    application.dependency_overrides[get_prediction_repo] = lambda: prediction_repo
    application.dependency_overrides[get_model_repo] = lambda: model_repo
    application.dependency_overrides[get_aemet_service] = lambda: aemet
    application.dependency_overrides[get_city_repo] = lambda: city_repo
    application.dependency_overrides[get_geo_service] = lambda: geo
    return application


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client
