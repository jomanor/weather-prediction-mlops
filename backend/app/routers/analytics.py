"""Analytics endpoints: ``/api/analytics/*``.

Both routes are read-only and share the process TTL cache
(``CACHE_TTL_AGGREGATE_SECONDS`` = 300 s -> ETag / ``Cache-Control``) via the
existing ``cached()`` helper. The window is part of
the cache key, so ``days`` changes never serve a stale window.
"""

import asyncio
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import TypeVar

from fastapi import APIRouter, Depends, Query, Request, Response

from app.core.cache import CACHE_TTL_AGGREGATE_SECONDS, TTLCache, cached, get_cache
from app.repositories.analytics_repo import AnalyticsRepository, get_analytics_repo
from app.routers.cities import MAX_NAME_LENGTH
from app.schemas.analytics import DiurnalResponse, WindRoseResponse

router = APIRouter(prefix="/analytics", tags=["analytics"])

#: Render free tier = one uvicorn worker / 0.1 CPU. The climate views fire
#: several cold queries at once; bound their computation so a burst cannot
#: saturate the loop. Two is enough to overlap Mongo I/O without piling up
#: Python-side aggregation on a single core.
ANALYTICS_MAX_CONCURRENCY = 2
_analytics_semaphore = asyncio.Semaphore(ANALYTICS_MAX_CONCURRENCY)

T = TypeVar("T")


async def _bounded(awaitable: Awaitable[T]) -> T:
    """Run one analytics repo call under the shared compute semaphore."""
    async with _analytics_semaphore:
        return await awaitable


CITY_QUERY = Query(..., min_length=1, max_length=MAX_NAME_LENGTH, description="City name")
WIND_DAYS_QUERY = Query(default=90, ge=7, le=365, description="Lookback window in days (7-365)")
DIURNAL_DAYS_QUERY = Query(default=90, ge=7, le=365, description="Lookback window in days (7-365)")


@router.get("/wind-rose", response_model=WindRoseResponse)
async def analytics_wind_rose(
    request: Request,
    response: Response,
    city: str = CITY_QUERY,
    days: int = WIND_DAYS_QUERY,
    repo: AnalyticsRepository = Depends(get_analytics_repo),
    cache: TTLCache = Depends(get_cache),
) -> WindRoseResponse | Response:
    """16-sector wind rose, count and mean speed in km/h."""
    now = datetime.now(timezone.utc)
    return await cached(
        request,
        response,
        cache,
        f"analytics:wind-rose:{city}:{days}",
        CACHE_TTL_AGGREGATE_SECONDS,
        lambda: _bounded(repo.wind_rose(city, days, now)),
    )


@router.get("/diurnal", response_model=DiurnalResponse)
async def analytics_diurnal(
    request: Request,
    response: Response,
    days: int = DIURNAL_DAYS_QUERY,
    repo: AnalyticsRepository = Depends(get_analytics_repo),
    cache: TTLCache = Depends(get_cache),
) -> DiurnalResponse | Response:
    """All canonical cities x local hour 0..23 mean temperature."""
    now = datetime.now(timezone.utc)
    return await cached(
        request,
        response,
        cache,
        f"analytics:diurnal:{days}",
        CACHE_TTL_AGGREGATE_SECONDS,
        lambda: _bounded(repo.diurnal(days, now)),
    )
