"""Weather endpoints: ``/api/weather/*``."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.repositories.weather_repo import WeatherRepository, get_weather_repo
from app.schemas.weather import (
    CurrentWeather,
    CurrentWeatherList,
    HistoryResponse,
    WeatherStats,
)
from app.services.stats import compute_stats

router = APIRouter(prefix="/weather", tags=["weather"])

HOURS_QUERY = Query(default=24, ge=1, le=168, description="Hours of data (max 168 = 1 week)")


@router.get("/current", response_model=CurrentWeatherList)
async def current_weather(
    repo: WeatherRepository = Depends(get_weather_repo),
) -> CurrentWeatherList:
    stations = await repo.latest_per_city()
    return CurrentWeatherList(count=len(stations), stations=stations)


@router.get("/current/{city}", response_model=CurrentWeather)
async def current_weather_for_city(
    city: str, repo: WeatherRepository = Depends(get_weather_repo)
) -> CurrentWeather:
    station = await repo.latest_for_city(city)
    if station is None:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    return station


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
    return HistoryResponse(city=city, hours=hours, count=len(points), points=points)


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
