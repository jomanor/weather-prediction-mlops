"""MongoDB access for the read-only analytics endpoints (``/api/analytics/*``).

All six endpoints read ``weather_features`` (plus ``weather_predictions`` for the
error-by-hour join). ``weather_features`` is a single overwritten snapshot bounded
by the 180-day ``raw_weather`` TTL (~4.3k hourly rows per city), so every read is
a per-city, index-backed query on the existing ``{city: 1, timestamp: -1}`` index
with a ``city``-prefixed sort — no aggregate pipeline is needed, hence there is no
``allowDiskUse`` to pass here. The prediction read is backed by
``{city: 1, prediction_timestamp: -1}``. The computations run in Python over a
bounded window (<= 365 * 24 rows per city), the same shape as
``WeatherRepository.quality_report``.

Wind speeds in ``weather_features`` are stored in m/s (the Open-Meteo writers
request ``wind_speed_unit=ms``); the contract promises km/h, so the conversion
happens here, at the boundary, and nowhere else.
"""

from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.cities import DEFAULT_CITIES
from app.core.coerce import as_utc, to_float, to_int
from app.db.mongo import get_db
from app.schemas.analytics import (
    ClimatologyPoint,
    ClimatologyResponse,
    CorrelationResponse,
    DailyPoint,
    DailyResponse,
    DiurnalCell,
    DiurnalResponse,
    ErrorByHourPoint,
    ErrorByHourResponse,
    WindRoseResponse,
    WindSector,
)

#: Stored m/s -> contract km/h.
MS_TO_KMH = 3.6
#: Degree-day base temperature, per Contract 4.
HDD_CDD_BASE = 18.0
#: A heatwave is >= 3 consecutive days with tmax >= 35 °C.
HEATWAVE_TMAX = 35.0
HEATWAVE_MIN_DAYS = 3
#: A day-of-year climatology needs at least this many distinct years to be honest.
MIN_CLIMATOLOGY_YEARS = 2
#: Pearson needs at least this many shared daily means, per Contract 4.
MIN_CORRELATION_DAYS = 3
#: 16 compass sectors, clockwise from North.
WIND_SECTORS = 16
WIND_SECTOR_DEGREES = 360.0 / WIND_SECTORS  # 22.5°

#: The fixed 14-station registry (sorted); diurnal/correlation report all of them
#: even when a city has no rows, so an outage is visible instead of silent.
CANONICAL_CITIES: tuple[str, ...] = tuple(sorted(name for name, _, _ in DEFAULT_CITIES))


def _mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length samples, or None.

    None when there are fewer than ``MIN_CORRELATION_DAYS`` shared days or when
    either sample has zero variance (the coefficient is undefined).
    """
    n = len(xs)
    if n < MIN_CORRELATION_DAYS or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x <= 0 or variance_y <= 0:
        return None
    rho = covariance / math.sqrt(variance_x * variance_y)
    # Guard against 1.0000000000000002 from floating-point error.
    return max(-1.0, min(1.0, rho))


def _error_stats(errors: list[float]) -> tuple[float | None, float | None, float | None]:
    """(mae, rmse, bias) for a list of signed errors; all None when empty."""
    n = len(errors)
    if n == 0:
        return None, None, None
    mae = sum(abs(error) for error in errors) / n
    rmse = math.sqrt(sum(error * error for error in errors) / n)
    bias = sum(errors) / n
    return mae, rmse, bias


def _wind_sector(direction: float | None) -> int | None:
    """``round(dir / 22.5) % 16``; None when the direction is missing."""
    if direction is None:
        return None
    return int(round(direction / WIND_SECTOR_DEGREES)) % WIND_SECTORS


def _heatwave_dates(dates: list[str], tmax_by_date: dict[str, float | None]) -> set[str]:
    """Dates that belong to a run of >= 3 consecutive days with tmax >= 35 °C.

    ``dates`` must list every calendar day in the window in ascending order, so a
    run in the list is a run in calendar time; a missing/None tmax breaks it.
    """
    flagged: set[str] = set()
    run: list[str] = []
    for day in dates:
        tmax = tmax_by_date.get(day)
        if tmax is not None and tmax >= HEATWAVE_TMAX:
            run.append(day)
            continue
        if len(run) >= HEATWAVE_MIN_DAYS:
            flagged.update(run)
        run = []
    if len(run) >= HEATWAVE_MIN_DAYS:
        flagged.update(run)
    return flagged


class AnalyticsRepository:
    """Read-only analytics over ``weather_features`` / ``weather_predictions``."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    @property
    def _features(self):
        return self._db["weather_features"]

    @property
    def _predictions(self):
        return self._db["weather_predictions"]

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

    async def daily(self, city: str, days: int, now: datetime) -> DailyResponse:
        """Per-day tmin/tmax/tmean, HDD/CDD, climatology anomaly and heatwaves."""
        end_date = now.date()
        start_date = end_date - timedelta(days=days - 1)
        window_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)

        rows = await self._read_features(city, ["temperature"])

        # Day-of-year climatology over all available history for the city.
        by_doy: dict[int, list[float]] = {}
        years_by_doy: dict[int, set[int]] = {}
        # Daily aggregates inside the window.
        by_date: dict[str, list[float]] = {}
        for doc in rows:
            timestamp = as_utc(doc.get("timestamp"))
            temperature = to_float(doc.get("temperature"))
            if timestamp is None:
                continue
            if temperature is not None:
                doy = timestamp.timetuple().tm_yday
                by_doy.setdefault(doy, []).append(temperature)
                years_by_doy.setdefault(doy, set()).add(timestamp.year)
            if window_start <= timestamp <= now and temperature is not None:
                by_date.setdefault(timestamp.date().isoformat(), []).append(temperature)

        climatology_mean = {doy: sum(values) / len(values) for doy, values in by_doy.items()}
        dates = [(start_date + timedelta(days=offset)).isoformat() for offset in range(days)]
        tmax_by_date = {day: max(values) for day, values in by_date.items()}
        heatwave_dates = _heatwave_dates(dates, tmax_by_date)

        points: list[DailyPoint] = []
        for day in dates:
            temperatures = by_date.get(day)
            tmin = min(temperatures) if temperatures else None
            tmax = max(temperatures) if temperatures else None
            tmean = _mean(temperatures) if temperatures else None
            hdd = max(0.0, HDD_CDD_BASE - tmean) if tmean is not None else None
            cdd = max(0.0, tmean - HDD_CDD_BASE) if tmean is not None else None
            doy = date.fromisoformat(day).timetuple().tm_yday
            climatology = climatology_mean.get(doy)
            enough_years = len(years_by_doy.get(doy, ())) >= MIN_CLIMATOLOGY_YEARS
            anomaly = (
                tmean - climatology
                if tmean is not None and climatology is not None and enough_years
                else None
            )
            points.append(
                DailyPoint(
                    date=day,
                    tmin=tmin,
                    tmax=tmax,
                    tmean=tmean,
                    hdd=hdd,
                    cdd=cdd,
                    anomaly=anomaly,
                    heatwave=day in heatwave_dates,
                )
            )

        return DailyResponse(city=city, days=days, generated_at=now, points=points)

    async def climatology(self, city: str, now: datetime) -> ClimatologyResponse:
        """Day-of-year 1..366 means for a city, with the distinct-year basis."""
        rows = await self._read_features(city, ["temperature"])
        by_doy: dict[int, list[float]] = {}
        years: set[int] = set()
        for doc in rows:
            timestamp = as_utc(doc.get("timestamp"))
            temperature = to_float(doc.get("temperature"))
            if timestamp is None:
                continue
            years.add(timestamp.year)
            if temperature is None:
                continue
            by_doy.setdefault(timestamp.timetuple().tm_yday, []).append(temperature)

        series = []
        for doy in range(1, 367):
            temperatures = by_doy.get(doy)
            series.append(
                ClimatologyPoint(
                    day_of_year=doy,
                    tmean=_mean(temperatures) if temperatures else None,
                    tmin=min(temperatures) if temperatures else None,
                    tmax=max(temperatures) if temperatures else None,
                    n=len(temperatures) if temperatures else 0,
                )
            )
        return ClimatologyResponse(
            city=city, generated_at=now, basis_years=float(len(years)), series=series
        )

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
        """All canonical cities x local hour 0..23, mean temperature per cell."""
        start = now - timedelta(days=days)
        per_city = await asyncio.gather(
            *(
                self._read_features(city, ["temperature", "hour"], start=start, end=now)
                for city in CANONICAL_CITIES
            )
        )

        cells: list[DiurnalCell] = []
        for city, rows in zip(CANONICAL_CITIES, per_city):
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

    async def correlation(self, days: int, var: str, now: datetime) -> CorrelationResponse:
        """Pairwise Pearson of daily city means, aligned by date, over the window."""
        start = now - timedelta(days=days)
        per_city = await asyncio.gather(
            *(self._read_features(city, [var], start=start, end=now) for city in CANONICAL_CITIES)
        )

        daily: list[dict[str, float]] = []
        for rows in per_city:
            by_date: dict[str, list[float]] = {}
            for doc in rows:
                timestamp = as_utc(doc.get("timestamp"))
                value = to_float(doc.get(var))
                if timestamp is None or value is None:
                    continue
                by_date.setdefault(timestamp.date().isoformat(), []).append(value)
            daily.append({day: sum(values) / len(values) for day, values in by_date.items()})

        cities = list(CANONICAL_CITIES)
        matrix: list[list[float | None]] = []
        for left in range(len(cities)):
            row: list[float | None] = []
            for right in range(len(cities)):
                shared = sorted(set(daily[left]) & set(daily[right]))
                row.append(
                    _pearson(
                        [daily[left][day] for day in shared],
                        [daily[right][day] for day in shared],
                    )
                )
            matrix.append(row)
        return CorrelationResponse(
            days=days, var=var, generated_at=now, cities=cities, matrix=matrix
        )

    async def error_by_hour(self, days: int, now: datetime) -> ErrorByHourResponse:
        """MAE/RMSE/bias per (horizon_hours, local hour) vs the observed target.

        The prediction's target time is ``source_timestamp + horizon_hours`` (the
        value the model actually predicts; ``prediction_timestamp`` is the Spark
        run time, not the target). Predictions whose target observation is
        missing — or whose target observation carries no local ``hour`` — are
        excluded, per Contract 4.
        """
        start = now - timedelta(days=days)
        per_city = await asyncio.gather(
            *(self._error_rows_for_city(city, start, now) for city in CANONICAL_CITIES)
        )

        errors: dict[tuple[int, int], list[float]] = {}
        persistence: dict[tuple[int, int], list[float]] = {}
        for rows in per_city:
            for horizon, hour, error, persist in rows:
                key = (horizon, hour)
                errors.setdefault(key, []).append(error)
                if persist is not None:
                    persistence.setdefault(key, []).append(persist)

        points: list[ErrorByHourPoint] = []
        for horizon, hour in sorted(errors):
            mae, rmse, bias = _error_stats(errors[(horizon, hour)])
            persistence_values = persistence.get((horizon, hour), [])
            points.append(
                ErrorByHourPoint(
                    horizon_hours=horizon,
                    hour=hour,
                    n=len(errors[(horizon, hour)]),
                    mae=mae,
                    rmse=rmse,
                    bias=bias,
                    persistence_mae=_mean([abs(value) for value in persistence_values]),
                )
            )
        return ErrorByHourResponse(days=days, generated_at=now, points=points)

    async def _error_rows_for_city(
        self, city: str, start: datetime, now: datetime
    ) -> list[tuple[int, int, float, float | None]]:
        """(horizon, local hour, error, persistence error) rows for one city."""
        cursor = self._predictions.find(
            {"city": city, "prediction_timestamp": {"$gte": start, "$lte": now}},
            {
                "city": 1,
                "source_timestamp": 1,
                "horizon_hours": 1,
                "predicted_temperature": 1,
                "observed_temperature": 1,
            },
        ).sort([("city", 1), ("prediction_timestamp", -1)])
        predictions = [doc async for doc in cursor]

        features = await self._read_features(city, ["temperature", "hour"])
        # Bucket the hourly observations by wall-clock hour: feature timestamps
        # are not minute-aligned (ingest runs at :50), so an exact-timestamp join
        # would silently drop almost every row.
        observed_by_hour: dict[datetime, tuple[float | None, int | None]] = {}
        for doc in features:
            timestamp = as_utc(doc.get("timestamp"))
            if timestamp is None:
                continue
            bucket = timestamp.replace(minute=0, second=0, microsecond=0)
            observed_by_hour[bucket] = (to_float(doc.get("temperature")), to_int(doc.get("hour")))

        rows: list[tuple[int, int, float, float | None]] = []
        for doc in predictions:
            source = as_utc(doc.get("source_timestamp"))
            horizon = to_int(doc.get("horizon_hours"))
            predicted = to_float(doc.get("predicted_temperature"))
            if source is None or horizon is None or predicted is None:
                continue
            target = (source + timedelta(hours=horizon)).replace(minute=0, second=0, microsecond=0)
            observation = observed_by_hour.get(target)
            if observation is None:
                continue
            observed, hour = observation
            if observed is None or hour is None:
                continue
            baseline = to_float(doc.get("observed_temperature"))
            persistence_error = observed - baseline if baseline is not None else None
            rows.append((horizon, hour, predicted - observed, persistence_error))
        return rows


def get_analytics_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> AnalyticsRepository:
    return AnalyticsRepository(db)
