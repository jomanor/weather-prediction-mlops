"""MongoDB access for ``weather_data`` and ``raw_weather``.

Two writers produce observations:

* ``weather_data`` — the Kafka consumer's current-conditions snapshot, nested
  under ``data.main`` / ``data.wind`` / ``data.clouds`` / ``data.weather``.
* ``raw_weather`` — the full Open-Meteo payload (consumer *and* the historical
  backfill script), with the per-hour values under ``payload.current``.

``weather_data`` is preferred; ``raw_weather`` is the fallback so historical
backfills are queryable too. One mapping helper per collection, no duplication.
"""

from datetime import datetime
from typing import Any

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.coerce import as_utc, first_not_none, to_float, to_int
from app.db.mongo import get_db
from app.schemas.weather import CurrentWeather


def _from_weather_data(doc: dict[str, Any]) -> CurrentWeather | None:
    """``weather_data`` document -> CurrentWeather."""
    observed_at = as_utc(doc.get("timestamp"))
    if observed_at is None:
        return None

    data = doc.get("data") or {}
    main = data.get("main") or {}
    wind = data.get("wind") or {}
    clouds = data.get("clouds") or {}
    weather = data.get("weather") or []
    weather_code = weather[0].get("id") if weather and isinstance(weather[0], dict) else None

    return CurrentWeather(
        city=doc.get("city", "unknown"),
        latitude=to_float(doc.get("latitude")),
        longitude=to_float(doc.get("longitude")),
        temperature=to_float(first_not_none(doc.get("temperature"), main.get("temp"))),
        apparent_temperature=to_float(
            first_not_none(doc.get("feels_like"), main.get("feels_like"))
        ),
        humidity=to_float(first_not_none(doc.get("humidity"), main.get("humidity"))),
        pressure=to_float(first_not_none(doc.get("pressure"), main.get("pressure"))),
        wind_speed=to_float(first_not_none(doc.get("wind_speed"), wind.get("speed"))),
        wind_direction=to_float(first_not_none(doc.get("wind_direction"), wind.get("deg"))),
        precipitation=to_float(data.get("precipitation")),
        cloud_cover=to_float(clouds.get("all")),
        weather_code=to_int(weather_code),
        observed_at=observed_at,
    )


def _from_raw_weather(doc: dict[str, Any]) -> CurrentWeather | None:
    """``raw_weather`` document -> CurrentWeather."""
    observed_at = as_utc(doc.get("timestamp"))
    if observed_at is None:
        return None

    payload = doc.get("payload") or {}
    current = payload.get("current") or {}

    return CurrentWeather(
        city=doc.get("city", payload.get("city", "unknown")),
        latitude=to_float(payload.get("latitude")),
        longitude=to_float(payload.get("longitude")),
        temperature=to_float(current.get("temperature_2m")),
        apparent_temperature=to_float(current.get("apparent_temperature")),
        humidity=to_float(current.get("relative_humidity_2m")),
        pressure=to_float(current.get("surface_pressure")),
        wind_speed=to_float(current.get("wind_speed_10m")),
        wind_direction=to_float(current.get("wind_direction_10m")),
        precipitation=to_float(current.get("precipitation")),
        cloud_cover=to_float(current.get("cloud_cover")),
        weather_code=to_int(current.get("weather_code")),
        observed_at=observed_at,
    )


def _map_all(docs: list[dict[str, Any]], mapper) -> list[CurrentWeather]:
    return [point for point in (mapper(doc) for doc in docs) if point is not None]


class WeatherRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    @property
    def _current(self):
        return self._db["weather_data"]

    @property
    def _raw(self):
        return self._db["raw_weather"]

    async def list_cities(self) -> list[str]:
        cities = set(await self._current.distinct("city"))
        cities.update(await self._raw.distinct("city"))
        return sorted(city for city in cities if city)

    async def latest_for_city(self, city: str) -> CurrentWeather | None:
        doc = await self._current.find_one({"city": city}, sort=[("timestamp", -1)])
        if doc is not None:
            return _from_weather_data(doc)

        doc = await self._raw.find_one({"city": city}, sort=[("timestamp", -1)])
        return _from_raw_weather(doc) if doc is not None else None

    async def latest_per_city(self) -> list[CurrentWeather]:
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {"_id": "$city", "latest": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$latest"}},
            {"$sort": {"city": 1}},
        ]
        docs = [doc async for doc in self._current.aggregate(pipeline)]
        if docs:
            return _map_all(docs, _from_weather_data)

        docs = [doc async for doc in self._raw.aggregate(pipeline)]
        return _map_all(docs, _from_raw_weather)

    async def find_range(
        self,
        city: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
        newest_first: bool = True,
    ) -> list[CurrentWeather]:
        query = {"city": city, "timestamp": {"$gte": start, "$lte": end}}
        direction = -1 if newest_first else 1

        cursor = self._current.find(query).sort("timestamp", direction).limit(limit)
        docs = [doc async for doc in cursor]
        if docs:
            return _map_all(docs, _from_weather_data)

        cursor = self._raw.find(query).sort("timestamp", direction).limit(limit)
        docs = [doc async for doc in cursor]
        return _map_all(docs, _from_raw_weather)


def get_weather_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> WeatherRepository:
    return WeatherRepository(db)
