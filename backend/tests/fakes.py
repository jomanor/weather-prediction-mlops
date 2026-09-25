"""In-memory stand-ins for the repositories and the AEMET service.

Used through FastAPI dependency overrides, so the whole suite runs offline.
"""

from datetime import datetime

from app.repositories.weather_repo import _downsample_by_step
from app.schemas.city import City, GeoResult
from app.schemas.models import ModelInfo
from app.schemas.predictions import Prediction
from app.schemas.weather import CurrentWeather
from app.services.aemet import AemetForecast
from app.services.geo import GeocodingError


class FakeMongoCursor:
    """Async cursor double that honours ``sort()``, ``limit()`` and ``projection``.

    A cursor that ignores these hides the two failure modes this suite exists to
    catch: a sort direction that does not match the index, and a projection that
    drops a field the mapper needs.
    """

    def __init__(self, documents: list[dict], projection: dict | None = None) -> None:
        self._documents = documents
        self._projection = projection
        self._sort: list[tuple[str, int]] = []
        self._limit: int | None = None
        #: Recorded so tests can assert the requested sort/index direction.
        self.sort_spec: list[tuple[str, int]] = []

    def sort(self, spec, direction: int | None = None):
        self._sort = [(spec, direction)] if direction is not None else list(spec)
        self.sort_spec = list(self._sort)
        return self

    def limit(self, count: int):
        self._limit = count
        return self

    def _resolved(self) -> list[dict]:
        documents = list(self._documents)
        # Stable multi-key sort: apply keys least-significant first.
        for key, direction in reversed(self._sort):
            documents.sort(key=lambda doc, k=key: doc.get(k), reverse=direction < 0)
        if self._limit is not None:
            documents = documents[: self._limit]
        return [_apply_projection(doc, self._projection) for doc in documents]

    def __aiter__(self):
        self._iterator = iter(self._resolved())
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None


def _apply_projection(document: dict, projection: dict | None) -> dict:
    """Apply a Mongo inclusion projection, preserving nested paths."""
    if not projection:
        return document
    projected: dict = {}
    for path, include in projection.items():
        if not include:
            continue
        value: object = document
        for part in path.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        if value is None:
            continue
        target = projected
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return projected


def _matches(document: dict, query: dict) -> bool:
    for key, expected in query.items():
        actual = document.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$gte" in expected and not (actual is not None and actual >= expected["$gte"]):
                return False
            if "$gt" in expected and not (actual is not None and actual > expected["$gt"]):
                return False
            if "$lte" in expected and not (actual is not None and actual <= expected["$lte"]):
                return False
        elif actual != expected:
            return False
    return True


class FakeMongoCollection:
    """Collection double: filters, then hands a cursor that sorts/projects."""

    def __init__(self, documents: list[dict] | None = None) -> None:
        self.documents = list(documents or [])
        self.find_calls: list[tuple[dict, dict | None]] = []
        self.cursors: list[FakeMongoCursor] = []

    def find(self, query, projection=None):
        self.find_calls.append((query, projection))
        cursor = FakeMongoCursor(
            [doc for doc in self.documents if _matches(doc, query)], projection
        )
        self.cursors.append(cursor)
        return cursor


class FakeMongoDb:
    """Minimal Motor database double for repository-level tests."""

    def __init__(
        self,
        weather_data: FakeMongoCollection | None = None,
        raw_weather: FakeMongoCollection | None = None,
    ) -> None:
        self._collections = {
            "weather_data": weather_data or FakeMongoCollection(),
            "raw_weather": raw_weather or FakeMongoCollection(),
        }

    def __getitem__(self, name):
        return self._collections[name]


class FakeWeatherRepository:
    def __init__(self, points: list[CurrentWeather] | None = None) -> None:
        self.points = list(points or [])
        #: Method names in call order, so tests can prove a cache hit skipped Mongo.
        self.calls: list[str] = []
        #: Last ``step_hours`` a bulk read was asked for, to prove query wiring.
        self.last_step_hours = 1

    async def list_cities(self) -> list[str]:
        self.calls.append("list_cities")
        return sorted({point.city for point in self.points})

    async def latest_for_city(self, city: str) -> CurrentWeather | None:
        self.calls.append("latest_for_city")
        matches = [point for point in self.points if point.city == city]
        return max(matches, key=lambda point: point.observed_at) if matches else None

    async def latest_per_city(self) -> list[CurrentWeather]:
        self.calls.append("latest_per_city")
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
        after: datetime | None = None,
    ) -> list[CurrentWeather]:
        self.calls.append("find_range")
        matches = [
            point
            for point in self.points
            if point.city == city
            and start <= point.observed_at <= end
            and (after is None or point.observed_at > after)
        ]
        matches.sort(key=lambda point: point.observed_at, reverse=newest_first)
        return matches[:limit]

    async def find_many_in_range(
        self,
        cities: list[str],
        start: datetime,
        end: datetime,
        fields: list[str] | None = None,
        step_hours: int = 1,
    ) -> list[CurrentWeather]:
        self.calls.append("find_many_in_range")
        self.last_step_hours = step_hours
        wanted = set(cities)
        matches = [
            point
            for point in self.points
            if point.city in wanted and start <= point.observed_at <= end
        ]
        matches.sort(key=lambda point: (point.city, point.observed_at))
        return _downsample_by_step(matches, step_hours)


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


class FakeCityRepository:
    """In-memory station registry with the same surface as ``CityRepository``."""

    def __init__(self, cities: list[City] | None = None) -> None:
        self.cities = list(cities or [])
        self.calls: list[str] = []

    async def list(self) -> list[City]:
        self.calls.append("list")
        return sorted(self.cities, key=lambda city: city.name)

    async def get(self, name: str) -> City | None:
        key = name.strip().casefold()
        for city in self.cities:
            if city.name.strip().casefold() == key:
                return city
        return None

    async def upsert(self, city: City) -> City:
        existing = await self.get(city.name)
        if existing is not None:
            self.cities.remove(existing)
        self.cities.append(city)
        return city

    async def delete(self, name: str) -> bool:
        existing = await self.get(name)
        if existing is None:
            return False
        self.cities.remove(existing)
        return True


class FakeDatabase:
    """Minimal Motor database stand-in for the readiness probe."""

    def __init__(self, name: str = "weather_db", error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.commands: list[dict] = []

    async def command(self, command: str, **kwargs) -> dict:
        self.commands.append({"command": command, **kwargs})
        if self.error is not None:
            raise self.error
        return {"ok": 1}


class FakeGeocodingService:
    """Same surface as ``GeocodingService`` but never touches the network."""

    def __init__(
        self,
        results: list[GeoResult] | None = None,
        error: str | None = None,
    ) -> None:
        self.results = list(results or [])
        self.error = error
        self.queries: list[str] = []

    async def search(self, query: str) -> list[GeoResult]:
        self.queries.append(query)
        if self.error is not None:
            raise GeocodingError(self.error)
        if not query.strip():
            return []
        return list(self.results)

    async def aclose(self) -> None:
        return None
