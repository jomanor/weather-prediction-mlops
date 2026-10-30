"""``POST /refresh``: optional, rate-limited server-side Open-Meteo fetch.

Writes one ``raw_weather`` document for the requested station. It never runs
Spark: whether features/predictions were refreshed is reported, always false.
The browser never triggers a workflow dispatch; this endpoint is the only
on-demand path and the process-local limiter (see ``app.core.rate_limit``)
keeps one caller from hammering the free Atlas tier.
"""

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings
from app.core.errors import APIError
from app.core.rate_limit import RefreshRateLimiter, get_refresh_limiter
from app.repositories.city_repo import CityRepository, get_city_repo
from app.repositories.weather_repo import WeatherRepository, get_weather_repo
from app.schemas.refresh import RefreshRequest, RefreshResponse
from app.services.openmeteo import (
    OpenMeteoError,
    OpenMeteoService,
    build_document,
    get_openmeteo_service,
)

router = APIRouter(tags=["refresh"])

#: Messages are deliberately generic: the submitted city name is echoed nowhere,
#: so an injected value cannot appear in a response body or a log line.
CITY_NOT_FOUND_MESSAGE = "City is not in the station registry"
RATE_LIMITED_MESSAGE = "Refresh rate limit exceeded"
UPSTREAM_ERROR_MESSAGE = "Open-Meteo request failed"
DISABLED_MESSAGE = "The refresh endpoint is disabled"


@router.post("/refresh", response_model=RefreshResponse)
async def refresh(
    body: RefreshRequest,
    request: Request,
    service: OpenMeteoService = Depends(get_openmeteo_service),
    limiter: RefreshRateLimiter = Depends(get_refresh_limiter),
    city_repo: CityRepository = Depends(get_city_repo),
    repo: WeatherRepository = Depends(get_weather_repo),
) -> RefreshResponse:
    settings: Settings = request.app.state.settings
    if not settings.refresh_enabled:
        raise APIError(503, "refresh_disabled", DISABLED_MESSAGE)

    # ``CityRepository.get`` is case-insensitive; compare the returned name to
    # enforce the exact match ingestion uses (and never fetch for "madrid").
    city = await city_repo.get(body.city)
    if city is None or city.name != body.city:
        raise APIError(404, "city_not_found", CITY_NOT_FOUND_MESSAGE)

    decision = await limiter.acquire(city.name)
    if not decision.allowed:
        raise APIError(
            429,
            "refresh_rate_limited",
            RATE_LIMITED_MESSAGE,
            headers={"Retry-After": str(decision.retry_after)},
        )

    try:
        data = await service.fetch_current(city.name, city.latitude, city.longitude)
    except OpenMeteoError as exc:
        raise APIError(502, "refresh_upstream_error", UPSTREAM_ERROR_MESSAGE) from exc

    document = build_document(data)
    if document is None:
        raise APIError(502, "refresh_upstream_error", UPSTREAM_ERROR_MESSAGE)

    await repo.save_raw_observation(document)
    return RefreshResponse(
        city=document["city"],
        source_timestamp=document["timestamp"],
        fetched_at=document["fetched_at"],
        features_refreshed=False,
        predictions_refreshed=False,
    )
