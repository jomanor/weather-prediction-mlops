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

#: Both writers request ``wind_speed_unit=ms`` (see ``scripts/ingest_weather.py``
#: and ``kafka/kafka-producer/producer.py``), so stored wind speeds are m/s;
#: the API contract (``CurrentWeather.wind_speed``) is km/h. Convert here, at
#: the boundary, and nowhere else.
MS_TO_KMH = 3.6


def _wind_kmh(value: Any) -> float | None:
    """Stored m/s -> contract km/h, preserving None for missing values."""
    speed = to_float(value)
    return speed * MS_TO_KMH if speed is not None else None


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
        wind_speed=_wind_kmh(first_not_none(doc.get("wind_speed"), wind.get("speed"))),
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
        wind_speed=_wind_kmh(current.get("wind_speed_10m")),
        wind_direction=to_float(current.get("wind_direction_10m")),
        precipitation=to_float(current.get("precipitation")),
        cloud_cover=to_float(current.get("cloud_cover")),
        weather_code=to_int(current.get("weather_code")),
        observed_at=observed_at,
    )


def _map_all(docs: list[dict[str, Any]], mapper) -> list[CurrentWeather]:
    return [point for point in (mapper(doc) for doc in docs) if point is not None]


#: Contract field -> stored paths, used to build a Mongo projection. Both the
#: top-level and nested spellings are listed because the two writers disagree;
#: ``_from_weather_data`` / ``_from_raw_weather`` already resolve either one.
_CURRENT_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature", "data.main.temp"),
    "apparent_temperature": ("feels_like", "data.main.feels_like"),
    "humidity": ("humidity", "data.main.humidity"),
    "pressure": ("pressure", "data.main.pressure"),
    "wind_speed": ("wind_speed", "data.wind.speed"),
    "wind_direction": ("wind_direction", "data.wind.deg"),
    "precipitation": ("data.precipitation",),
    "cloud_cover": ("data.clouds.all",),
    "weather_code": ("data.weather",),
}

_RAW_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "temperature": ("payload.current.temperature_2m",),
    "apparent_temperature": ("payload.current.apparent_temperature",),
    "humidity": ("payload.current.relative_humidity_2m",),
    "pressure": ("payload.current.surface_pressure",),
    "wind_speed": ("payload.current.wind_speed_10m",),
    "wind_direction": ("payload.current.wind_direction_10m",),
    "precipitation": ("payload.current.precipitation",),
    "cloud_cover": ("payload.current.cloud_cover",),
    "weather_code": ("payload.current.weather_code",),
}

#: Columns ``_from_weather_data`` / ``_from_raw_weather`` need regardless of the
#: requested projection: the two collections store the coordinates in different
#: places (top-level vs. nested under ``payload``), so the identity set is
#: collection-specific. Omitting them serializes ``latitude``/``longitude`` as
#: null whenever a projection is used.
_CURRENT_IDENTITY: tuple[str, ...] = ("city", "timestamp", "latitude", "longitude")
_RAW_IDENTITY: tuple[str, ...] = (
    "city",
    "timestamp",
    "payload.latitude",
    "payload.longitude",
)


def _projection(
    fields: list[str] | None,
    paths: dict[str, tuple[str, ...]],
    identity: tuple[str, ...],
):
    """Projection limited to ``fields`` plus ``identity`` columns, or None for all."""
    if not fields:
        return None
    projection: dict[str, int] = {column: 1 for column in identity}
    for field in fields:
        for path in paths.get(field, ()):
            projection[path] = 1
    return projection


def _downsample_by_step(points: list[CurrentWeather], step_hours: int) -> list[CurrentWeather]:
    """Keep the first observation per ``step_hours``-wide UTC bucket, per city."""
    if step_hours <= 1:
        return points
    bucket_seconds = step_hours * 3600
    kept: list[CurrentWeather] = []
    seen: set[tuple[str, int]] = set()
    for point in points:
        bucket = int(point.observed_at.timestamp()) // bucket_seconds
        if (point.city, bucket) in seen:
            continue
        seen.add((point.city, bucket))
        kept.append(point)
    return kept


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
        # Sort by (city, timestamp) so the ``{city: 1, timestamp: -1}`` index
        # feeds the group directly. Sorting on ``timestamp`` alone is a blocking
        # sort across the whole collection, which exceeds the 32 MB in-memory
        # limit on Atlas once the backfill fills it in. ``allowDiskUse`` keeps
        # the query from failing with code 292 even when the planner cannot use
        # an index (e.g. an index that has not been built yet).
        pipeline = [
            {"$sort": {"city": 1, "timestamp": -1}},
            {"$group": {"_id": "$city", "latest": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$latest"}},
            {"$sort": {"city": 1}},
        ]
        docs = [doc async for doc in self._current.aggregate(pipeline, allowDiskUse=True)]
        if docs:
            return _map_all(docs, _from_weather_data)

        docs = [doc async for doc in self._raw.aggregate(pipeline, allowDiskUse=True)]
        return _map_all(docs, _from_raw_weather)

    async def find_range(
        self,
        city: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
        newest_first: bool = True,
        after: datetime | None = None,
    ) -> list[CurrentWeather]:
        """Observations in ``[start, end]``, optionally strictly after ``after`` (keyset)."""
        window: dict[str, datetime] = {"$gte": start, "$lte": end}
        if after is not None:
            window["$gt"] = after
        query = {"city": city, "timestamp": window}
        direction = -1 if newest_first else 1

        cursor = self._current.find(query).sort("timestamp", direction).limit(limit)
        docs = [doc async for doc in cursor]
        if docs:
            return _map_all(docs, _from_weather_data)

        cursor = self._raw.find(query).sort("timestamp", direction).limit(limit)
        docs = [doc async for doc in cursor]
        return _map_all(docs, _from_raw_weather)

    async def find_many_in_range(
        self,
        cities: list[str],
        start: datetime,
        end: datetime,
        fields: list[str] | None = None,
        step_hours: int = 1,
    ) -> list[CurrentWeather]:
        """Bulk read for ``cities`` in ``[start, end]`` with one query per collection.

        ``weather_data`` is read first; any requested city without rows in range
        falls back to ``raw_weather`` (the repo's usual policy, applied per city
        here because the request spans cities). ``fields`` limits the Mongo
        projection to the requested contract fields — never to the query alone.

        Sorted ``(city asc, timestamp desc)`` to match the
        ``{city: 1, timestamp: -1}`` index, so the server walks the index with no
        blocking ``SORT`` stage (code 292). Output is ascending because the
        in-Python sort below reorders the points.
        """
        requested = list(dict.fromkeys(city for city in cities if city))
        if not requested:
            return []

        query = {"city": {"$in": requested}, "timestamp": {"$gte": start, "$lte": end}}
        sort: list[tuple[str, int]] = [("city", 1), ("timestamp", -1)]

        cursor = self._current.find(
            query, _projection(fields, _CURRENT_FIELD_PATHS, _CURRENT_IDENTITY)
        ).sort(sort)
        points = _map_all([doc async for doc in cursor], _from_weather_data)

        missing = [city for city in requested if city not in {point.city for point in points}]
        if missing:
            fallback = {
                "city": {"$in": missing},
                "timestamp": {"$gte": start, "$lte": end},
            }
            raw_cursor = self._raw.find(
                fallback, _projection(fields, _RAW_FIELD_PATHS, _RAW_IDENTITY)
            ).sort(sort)
            points.extend(_map_all([doc async for doc in raw_cursor], _from_raw_weather))

        points.sort(key=lambda point: (point.city, point.observed_at))
        return _downsample_by_step(points, step_hours)


def get_weather_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> WeatherRepository:
    return WeatherRepository(db)
