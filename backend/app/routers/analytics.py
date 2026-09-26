"""Analytics endpoints: ``/api/analytics/*`` (Batch 4, Contract 4).

All six routes are read-only and share the process TTL cache (300 s -> ETag /
``Cache-Control``) via the existing ``cached()`` helper. The window is part of
the cache key, so ``days`` changes never serve a stale window.
"""

import asyncio
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Literal, TypeVar

from fastapi import APIRouter, Depends, Query, Request, Response

from app.core.cache import TTLCache, cached, get_cache
from app.repositories.analytics_repo import AnalyticsRepository, get_analytics_repo
from app.routers.cities import MAX_NAME_LENGTH
from app.schemas.analytics import (
    ClimatologyResponse,
    CorrelationResponse,
    DailyResponse,
    DiurnalResponse,
    ErrorByHourResponse,
    WindRoseResponse,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])

ANALYTICS_TTL_SECONDS = 300

#: Render free tier = one uvicorn worker / 0.1 CPU. The analytics page fires
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
DAILY_DAYS_QUERY = Query(default=90, ge=7, le=180, description="Lookback window in days (7-180)")
WIND_DAYS_QUERY = Query(default=90, ge=7, le=365, description="Lookback window in days (7-365)")
DIURNAL_DAYS_QUERY = Query(default=90, ge=7, le=365, description="Lookback window in days (7-365)")
CORRELATION_DAYS_QUERY = Query(
    default=90, ge=7, le=365, description="Lookback window in days (7-365)"
)
ERROR_DAYS_QUERY = Query(default=30, ge=7, le=90, description="Lookback window in days (7-90)")
VAR_QUERY = Query(
    default="temperature",
    description="Numeric variable to correlate (temperature|humidity|pressure|wind_speed)",
)


@router.get("/daily", response_model=DailyResponse)
async def analytics_daily(
    request: Request,
    response: Response,
    city: str = CITY_QUERY,
    days: int = DAILY_DAYS_QUERY,
    repo: AnalyticsRepository = Depends(get_analytics_repo),
    cache: TTLCache = Depends(get_cache),
) -> DailyResponse | Response:
    """Daily tmin/tmax/tmean, HDD/CDD, climatology anomaly and heatwave flags."""
    now = datetime.now(timezone.utc)
    return await cached(
        request,
        response,
        cache,
        f"analytics:daily:{city}:{days}",
        ANALYTICS_TTL_SECONDS,
        lambda: _bounded(repo.daily(city, days, now)),
    )


@router.get("/climatology", response_model=ClimatologyResponse)
async def analytics_climatology(
    request: Request,
    response: Response,
    city: str = CITY_QUERY,
    repo: AnalyticsRepository = Depends(get_analytics_repo),
    cache: TTLCache = Depends(get_cache),
) -> ClimatologyResponse | Response:
    """Day-of-year 1..366 climatology for one city."""
    now = datetime.now(timezone.utc)
    return await cached(
        request,
        response,
        cache,
        f"analytics:climatology:{city}",
        ANALYTICS_TTL_SECONDS,
        lambda: _bounded(repo.climatology(city, now)),
    )


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
        ANALYTICS_TTL_SECONDS,
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
        ANALYTICS_TTL_SECONDS,
        lambda: _bounded(repo.diurnal(days, now)),
    )


@router.get("/correlation", response_model=CorrelationResponse)
async def analytics_correlation(
    request: Request,
    response: Response,
    days: int = CORRELATION_DAYS_QUERY,
    var: Literal["temperature", "humidity", "pressure", "wind_speed"] = VAR_QUERY,
    repo: AnalyticsRepository = Depends(get_analytics_repo),
    cache: TTLCache = Depends(get_cache),
) -> CorrelationResponse | Response:
    """Pairwise Pearson of daily city means, aligned by date (null < 3 shared days)."""
    now = datetime.now(timezone.utc)
    return await cached(
        request,
        response,
        cache,
        f"analytics:correlation:{days}:{var}",
        ANALYTICS_TTL_SECONDS,
        lambda: _bounded(repo.correlation(days, var, now)),
    )


@router.get("/error-by-hour", response_model=ErrorByHourResponse)
async def analytics_error_by_hour(
    request: Request,
    response: Response,
    days: int = ERROR_DAYS_QUERY,
    repo: AnalyticsRepository = Depends(get_analytics_repo),
    cache: TTLCache = Depends(get_cache),
) -> ErrorByHourResponse | Response:
    """MAE/RMSE/bias/persistence-MAE per (horizon_hours, local hour)."""
    now = datetime.now(timezone.utc)
    return await cached(
        request,
        response,
        cache,
        f"analytics:error-by-hour:{days}",
        ANALYTICS_TTL_SECONDS,
        lambda: _bounded(repo.error_by_hour(days, now)),
    )
