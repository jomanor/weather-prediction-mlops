"""Weather endpoints: ``/api/weather/*``."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.core.cache import TTLCache, cached, get_cache
from app.repositories.weather_repo import WeatherRepository, get_weather_repo
from app.routers.cities import MAX_NAME_LENGTH
from app.schemas.weather import (
    WEATHER_FIELDS,
    CitySeries,
    CitySummary,
    CurrentWeather,
    CurrentWeatherList,
    HistoryResponse,
    SeriesResponse,
    SummaryResponse,
    WeatherPoint,
    WeatherRangeResponse,
    WeatherStats,
)
from app.services.stats import compute_stats

router = APIRouter(prefix="/weather", tags=["weather"])

HOURS_QUERY = Query(default=24, ge=1, le=168, description="Hours of data (max 168 = 1 week)")

CURRENT_TTL_SECONDS = 30
MAX_BULK_CITIES = 15
MAX_RANGE_LIMIT = 2000
MAX_SPARKLINE_POINTS = 48

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

    result = await cached(request, response, cache, "weather:current", CURRENT_TTL_SECONDS, load)
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
        request, response, cache, f"weather:current:{city}", CURRENT_TTL_SECONDS, load
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


@router.get("/range/{city}", response_model=WeatherRangeResponse)
async def weather_range(
    city: str,
    from_: datetime = Query(alias="from", description="Range start (ISO-8601 UTC)"),
    to: datetime = Query(alias="to", description="Range end (ISO-8601 UTC)"),
    limit: int = Query(default=1000, ge=1, le=MAX_RANGE_LIMIT, description="Page size"),
    cursor: datetime | None = Query(
        default=None, description="Keyset cursor: the last timestamp already returned"
    ),
    repo: WeatherRepository = Depends(get_weather_repo),
) -> WeatherRangeResponse:
    """Arbitrary window, oldest -> newest, keyset-paginated on ``timestamp``."""
    start, end = _as_utc(from_), _as_utc(to)
    if start >= end:
        raise HTTPException(status_code=422, detail="from must be earlier than to")

    after = _as_utc(cursor) if cursor is not None else None
    if after is not None and not start <= after <= end:
        raise HTTPException(status_code=422, detail="cursor must fall within [from, to]")

    points: list[WeatherPoint] = await repo.find_range(
        city,
        start,
        end,
        limit=limit,
        newest_first=False,
        after=after,
    )
    next_cursor = points[-1].observed_at if len(points) == limit else None
    return WeatherRangeResponse(
        city=city, count=len(points), points=points, next_cursor=next_cursor
    )


@router.get("/series", response_model=SeriesResponse)
async def weather_series(
    cities: str = CITIES_QUERY,
    hours: int = HOURS_QUERY,
    fields: str | None = FIELDS_QUERY,
    step_hours: int = STEP_HOURS_QUERY,
    repo: WeatherRepository = Depends(get_weather_repo),
) -> SeriesResponse:
    """One bulk query for up to 15 cities; one compact series per city."""
    names = _parse_cities(cities)
    selected = _parse_fields(fields)
    now = datetime.now(timezone.utc)
    points = await repo.find_many_in_range(
        names, now - timedelta(hours=hours), now, fields=selected, step_hours=step_hours
    )
    return SeriesResponse(
        hours=hours,
        count=len(names),
        cities=[_city_series(name, points) for name in names],
    )


def _city_series(name: str, points: list[WeatherPoint]) -> CitySeries:
    series = [point for point in points if point.city == name]
    return CitySeries(
        city=name,
        count=len(series),
        latest_timestamp=max((point.observed_at for point in series), default=None),
        points=series,
    )


@router.get("/summary", response_model=SummaryResponse)
async def weather_summary(
    cities: str = CITIES_QUERY,
    hours: int = HOURS_QUERY,
    repo: WeatherRepository = Depends(get_weather_repo),
) -> SummaryResponse:
    """Per-city min/max/first/last/trend plus a downsampled numeric sparkline."""
    names = _parse_cities(cities)
    now = datetime.now(timezone.utc)
    points = await repo.find_many_in_range(
        names, now - timedelta(hours=hours), now, fields=["temperature"]
    )
    return SummaryResponse(
        hours=hours,
        count=len(names),
        cities=[_city_summary(name, points) for name in names],
    )


def _city_summary(name: str, points: list[WeatherPoint]) -> CitySummary:
    values = [
        point.temperature
        for point in points
        if point.city == name and point.temperature is not None
    ]
    if not values:
        return CitySummary(city=name, points=[])
    return CitySummary(
        city=name,
        min=min(values),
        max=max(values),
        first=values[0],
        last=values[-1],
        trend=values[-1] - values[0],
        points=_downsample(values, MAX_SPARKLINE_POINTS),
    )


def _downsample(values: list[float], limit: int) -> list[float]:
    """Evenly spaced values, always including the first and last."""
    if len(values) <= limit:
        return list(values)
    step = (len(values) - 1) / (limit - 1)
    return [values[round(index * step)] for index in range(limit)]


@router.get("/stats/{city}", response_model=WeatherStats)
async def weather_stats(
    city: str,
    hours: int = HOURS_QUERY,
    repo: WeatherRepository = Depends(get_weather_repo),
) -> WeatherStats:
    now = datetime.now(timezone.utc)
    points = await repo.find_range(
        city, now - timedelta(hours=hours), now, limit=1000, newest_first=False
    )
    if not points:
        raise HTTPException(status_code=404, detail=f"No data found for city '{city}'")
    return compute_stats(city, hours, points)
