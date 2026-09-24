"""In-memory stand-ins for the repositories and the AEMET service.

Used through FastAPI dependency overrides, so the whole suite runs offline.
"""

from datetime import datetime

from app.schemas.models import ModelInfo
from app.schemas.predictions import Prediction
from app.schemas.weather import CurrentWeather
from app.services.aemet import AemetForecast


class FakeWeatherRepository:
    def __init__(self, points: list[CurrentWeather] | None = None) -> None:
        self.points = list(points or [])

    async def list_cities(self) -> list[str]:
        return sorted({point.city for point in self.points})

    async def latest_for_city(self, city: str) -> CurrentWeather | None:
        matches = [point for point in self.points if point.city == city]
        return max(matches, key=lambda point: point.observed_at) if matches else None

    async def latest_per_city(self) -> list[CurrentWeather]:
        latest: dict[str, CurrentWeather] = {}
        for point in self.points:
            current = latest.get(point.city)
            if current is None or point.observed_at > current.observed_at:
                latest[point.city] = point
        return [latest[city] for city in sorted(latest)]

    async def find_range(
        self,
        city: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
        newest_first: bool = True,
    ) -> list[CurrentWeather]:
        matches = [
            point
            for point in self.points
            if point.city == city and start <= point.observed_at <= end
        ]
        matches.sort(key=lambda point: point.observed_at, reverse=newest_first)
        return matches[:limit]


class FakePredictionRepository:
    def __init__(self, predictions: list[Prediction] | None = None) -> None:
        self.predictions = list(predictions or [])

    async def latest_per_city(self) -> list[Prediction]:
        latest: dict[str, Prediction] = {}
        for prediction in self.predictions:
            current = latest.get(prediction.city)
            if current is None or prediction.prediction_timestamp > current.prediction_timestamp:
                latest[prediction.city] = prediction
        return [latest[city] for city in sorted(latest)]

    async def for_city(self, city: str, limit: int = 48) -> list[Prediction]:
        matches = [p for p in self.predictions if p.city == city]
        matches.sort(key=lambda p: p.prediction_timestamp, reverse=True)
        return matches[:limit]

    async def find_range(self, city: str, start: datetime, end: datetime) -> list[Prediction]:
        matches = [
            p for p in self.predictions if p.city == city and start <= p.source_timestamp <= end
        ]
        matches.sort(key=lambda p: p.source_timestamp)
        return matches


class FakeModelRepository:
    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self.models = list(models or [])

    async def list_models(self, limit: int = 100) -> list[ModelInfo]:
        return self.models[:limit]


class FakeAemetService:
    """Same surface as ``AemetService`` but never touches the network."""

    def __init__(
        self,
        temps: dict[datetime, float] | None = None,
        error: str | None = None,
        issued_at: datetime | None = None,
        configured: bool = True,
    ) -> None:
        self.temps = dict(temps or {})
        self.error = error
        self.issued_at = issued_at
        self.configured = configured
        self.calls: list[str] = []

    async def forecast(self, city: str) -> AemetForecast:
        self.calls.append(city)
        if self.error is not None or not self.configured:
            return AemetForecast(
                available=False,
                error=self.error or "AEMET_API_KEY is not configured",
            )
        return AemetForecast(available=True, issued_at=self.issued_at, temps=dict(self.temps))

    async def aclose(self) -> None:
        return None
