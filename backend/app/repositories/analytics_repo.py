"""MongoDB access for the read-only analytics endpoints (``/api/analytics/*``).

Both endpoints read ``weather_features``. It is a single overwritten snapshot
bounded by the 180-day ``raw_weather`` TTL (~4.3k hourly rows per city), so every
read is a per-city, index-backed query on the existing
``{city: 1, timestamp: -1}`` index with a ``city``-prefixed sort — no aggregate
pipeline is needed, hence there is no ``allowDiskUse`` to pass here. The
computations run in Python over a bounded window (<= 365 * 24 rows per city), the
same shape as ``WeatherRepository.quality_report``.

Wind speeds in ``weather_features`` are stored in m/s (the Open-Meteo writers
request ``wind_speed_unit=ms``); the contract promises km/h, so the conversion
happens here, at the boundary, and nowhere else.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Iterable

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.coerce import to_float, to_int
from app.core.units import MS_TO_KMH
from app.db.mongo import get_db
from app.repositories.city_repo import registry_names
from app.schemas.analytics import DiurnalCell, DiurnalResponse, WindRoseResponse, WindSector

#: 16 compass sectors, clockwise from North.
WIND_SECTORS = 16
WIND_SECTOR_DEGREES = 360.0 / WIND_SECTORS  # 22.5°


def _mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _wind_sector(direction: float | None) -> int | None:
    """``round(dir / 22.5) % 16``; None when the direction is missing."""
    if direction is None:
        return None
    return int(round(direction / WIND_SECTOR_DEGREES)) % WIND_SECTORS


class AnalyticsRepository:
    """Read-only analytics over ``weather_features``."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    @property
    def _features(self):
        return self._db["weather_features"]

    async def _read_features(
        self,
        city: str,
        fields: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Per-city feature rows, oldest first, projected to ``fields``.

        The query is an equality on ``city`` plus a ``timestamp`` range, sorted
        ``[("city", 1), ("timestamp", 1)]`` so the ``{city: 1, timestamp: -1}``
        index feeds it directly (reverse scan) with no blocking sort.
        """
        query: dict[str, Any] = {"city": city}
        if start is not None or end is not None:
            window: dict[str, datetime] = {}
            if start is not None:
                window["$gte"] = start
            if end is not None:
                window["$lte"] = end
            query["timestamp"] = window
        projection: dict[str, int] = {"city": 1, "timestamp": 1}
        for field in fields:
            projection[field] = 1
        cursor = self._features.find(query, projection).sort([("city", 1), ("timestamp", 1)])
        return [doc async for doc in cursor]

    async def wind_rose(self, city: str, days: int, now: datetime) -> WindRoseResponse:
        """16-sector wind rose (count + mean speed in km/h) for the window."""
        start = now - timedelta(days=days)
        rows = await self._read_features(
            city, ["wind_direction", "wind_speed"], start=start, end=now
        )
        counts = [0] * WIND_SECTORS
        speeds: list[list[float]] = [[] for _ in range(WIND_SECTORS)]
        for doc in rows:
            sector = _wind_sector(to_float(doc.get("wind_direction")))
            if sector is None:
                continue
            counts[sector] += 1
            speed = to_float(doc.get("wind_speed"))
            if speed is not None:
                speeds[sector].append(speed * MS_TO_KMH)

        sectors = [
            WindSector(sector=index, count=counts[index], mean_speed=_mean(speeds[index]))
            for index in range(WIND_SECTORS)
        ]
        return WindRoseResponse(city=city, days=days, generated_at=now, sectors=sectors)

    async def diurnal(self, days: int, now: datetime) -> DiurnalResponse:
        """Every registry station x local hour 0..23, mean temperature per cell."""
        cities = await registry_names(self._db)
        start = now - timedelta(days=days)
        per_city = await asyncio.gather(
            *(
                self._read_features(city, ["temperature", "hour"], start=start, end=now)
                for city in cities
            )
        )

        cells: list[DiurnalCell] = []
        for city, rows in zip(cities, per_city):
            by_hour: dict[int, list[float]] = {}
            for doc in rows:
                hour = to_int(doc.get("hour"))
                temperature = to_float(doc.get("temperature"))
                if hour is None or temperature is None:
                    continue
                by_hour.setdefault(hour, []).append(temperature)
            for hour in range(24):
                temperatures = by_hour.get(hour)
                cells.append(
                    DiurnalCell(
                        city=city,
                        hour=hour,
                        tmean=_mean(temperatures) if temperatures else None,
                        n=len(temperatures) if temperatures else 0,
                    )
                )
        return DiurnalResponse(days=days, generated_at=now, cells=cells)


def get_analytics_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> AnalyticsRepository:
    return AnalyticsRepository(db)
