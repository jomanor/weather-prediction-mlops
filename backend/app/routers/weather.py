"""Weather endpoints: ``/api/weather/*``."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.core.cache import (
    CACHE_TTL_AGGREGATE_SECONDS,
    CACHE_TTL_LIVE_SECONDS,
    TTLCache,
    cached,
    get_cache,
)
from app.repositories.weather_repo import WeatherRepository, get_weather_repo
from app.routers.cities import MAX_NAME_LENGTH
from app.schemas.quality import WeatherQuality
from app.schemas.weather import (
    WEATHER_FIELDS,
    CitySeries,
    CurrentWeather,
    CurrentWeatherList,
    HistoryResponse,
    SeriesResponse,
    WeatherPoint,
)

router = APIRouter(prefix="/weather", tags=["weather"])

HOURS_QUERY = Query(default=24, ge=1, le=168, description="Hours of data (max 168 = 1 week)")

MAX_BULK_CITIES = 15

CITIES_QUERY = Query(
    ...,
    description="Comma-separated city names (1-15)",
)
FIELDS_QUERY = Query(
    default=None,
    description="Comma-separated weather fields to project (defaults to all)",
)
STEP_HOURS_QUERY = Query(
    default=1,
    ge=1,
    le=24,
    description="Downsample to at most one point per N hours (1-24)",
)
DAYS_QUERY = Query(default=7, ge=1, le=30, description="Quality lookback window in days (1-30)")


def _parse_cities(raw: str) -> list[str]:
    names = list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not 1 <= len(names) <= MAX_BULK_CITIES:
        raise HTTPException(
            status_code=422,
            detail=f"cities must contain between 1 and {MAX_BULK_CITIES} names",
        )
    # Same cap the registry applies to a city name (reused, not re-declared):
    # an unbounded name would go straight into the ``$in``.
    if any(len(name) > MAX_NAME_LENGTH for name in names):
        raise HTTPException(
            status_code=422,
            detail=f"city names must be at most {MAX_NAME_LENGTH} characters",
        )
    return names


def _parse_fields(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    fields = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(fields) - WEATHER_FIELDS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown field(s): {', '.join(unknown)}",
        )
    return fields or None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _data_age_seconds(observed_at: datetime | None) -> int | None:
    if observed_at is None:
        return None
    return max(0, int((datetime.now(timezone.utc) - _as_utc(observed_at)).total_seconds()))


def _set_data_age(response: Response, observed_at: datetime | None) -> None:
    age = _data_age_seconds(observed_at)
    if age is not None:
        response.headers["X-Data-Age-Seconds"] = str(age)


@router.get("/current", response_model=CurrentWeatherList)
async def current_weather(
    request: Request,
    response: Response,
    repo: WeatherRepository = Depends(get_weather_repo),
    cache: TTLCache = Depends(get_cache),
) -> CurrentWeatherList | Response:
    async def load() -> CurrentWeatherList:
        stations = await repo.latest_per_city()
        return CurrentWeatherList(count=len(stations), stations=stations)

    result = await cached(request, response, cache, "weather:current", CACHE_TTL_LIVE_SECONDS, load)
    if isinstance(result, Response):
        return result
    _set_data_age(
        response,
        max((station.observed_at for station in result.stations), default=None),
    )
    return result


@router.get("/current/{city}", response_model=CurrentWeather)
async def current_weather_for_city(
    city: str,
    request: Request,
    response: Response,
    repo: WeatherRepository = Depends(get_weather_repo),
    cache: TTLCache = Depends(get_cache),
) -> CurrentWeather | Response:
    async def load() -> CurrentWeather:
        station = await repo.latest_for_city(city)
        if station is None:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
        return station

    result = await cached(
        request, response, cache, f"weather:current:{city}", CACHE_TTL_LIVE_SECONDS, load
    )
    if isinstance(result, Response):
        return result
    _set_data_age(response, result.observed_at)
    return result


@router.get("/history/{city}", response_model=HistoryResponse)
async def weather_history(
    city: str,
    hours: int = HOURS_QUERY,
    limit: int = Query(default=200, ge=1, le=1000, description="Maximum number of records"),
    repo: WeatherRepository = Depends(get_weather_repo),
) -> HistoryResponse:
    now = datetime.now(timezone.utc)
    points = await repo.find_range(
        city, now - timedelta(hours=hours), now, limit=limit, newest_first=True
    )
    if not points:
        raise HTTPException(status_code=404, detail=f"No historical data found for city '{city}'")
    # Query newest-first so ``limit`` keeps the most recent window, then flip to
    # chronological order: charts read left-to-right from oldest to newest.
    return HistoryResponse(city=city, hours=hours, count=len(points), points=list(reversed(points)))


@router.get("/series", response_model=SeriesResponse)
async def weather_series(
    request: Request,
    response: Response,
    cities: str = CITIES_QUERY,
    hours: int = HOURS_QUERY,
    fields: str | None = FIELDS_QUERY,
    step_hours: int = STEP_HOURS_QUERY,
    repo: WeatherRepository = Depends(get_weather_repo),
    cache: TTLCache = Depends(get_cache),
) -> SeriesResponse | Response:
    """One bulk query for up to 15 cities; one compact series per city."""
    names = _parse_cities(cities)
    selected = _parse_fields(fields)
    key = _series_key(names, hours, selected, step_hours)

    async def load() -> SeriesResponse:
        now = datetime.now(timezone.utc)
        points = await repo.find_many_in_range(
            names, now - timedelta(hours=hours), now, fields=selected, step_hours=step_hours
        )
        return SeriesResponse(
            hours=hours,
            count=len(names),
            cities=[_city_series(name, points) for name in names],
        )

    return await cached(request, response, cache, key, CACHE_TTL_LIVE_SECONDS, load)


def _series_key(names: list[str], hours: int, fields: list[str] | None, step_hours: int) -> str:
    """Cache key covering every parameter that changes the ``/series`` payload.

    ``names`` keeps the request order (it is the order of ``cities`` in the
    response) and is already stripped/deduplicated by ``_parse_cities``. The
    projected ``fields`` are sorted: the projection is a set, so a reordered
    field list produces the same payload and must hit the same entry.
    """
    return (
        f"weather:series:{','.join(names)}"
        f":{hours}:{','.join(sorted(fields)) if fields else 'all'}:{step_hours}"
    )


def _city_series(name: str, points: list[WeatherPoint]) -> CitySeries:
    series = [point for point in points if point.city == name]
    return CitySeries(
        city=name,
        count=len(series),
        latest_timestamp=max((point.observed_at for point in series), default=None),
        points=series,
    )


@router.get("/quality", response_model=WeatherQuality)
async def weather_quality(
    request: Request,
    response: Response,
    days: int = DAYS_QUERY,
    repo: WeatherRepository = Depends(get_weather_repo),
    cache: TTLCache = Depends(get_cache),
) -> WeatherQuality | Response:
    """Per-city completeness/gap/null/age meter over ``weather_features``.

    Cached with the shared helper so it gets an ETag and ``Cache-Control`` like
    every other read route; the window is part of the cache key.
    """
    now = datetime.now(timezone.utc)

    async def load() -> WeatherQuality:
        return WeatherQuality(
            generated_at=now,
            days=days,
            cities=await repo.quality_report(days, now),
        )

    return await cached(
        request, response, cache, f"weather:quality:{days}", CACHE_TTL_AGGREGATE_SECONDS, load
    )
