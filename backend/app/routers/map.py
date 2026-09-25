"""Map endpoints: ``GET /api/map/stations`` as a GeoJSON FeatureCollection."""

from fastapi import APIRouter, Depends, Request, Response

from app.core.cache import TTLCache, cached, get_cache
from app.repositories.weather_repo import WeatherRepository, get_weather_repo
from app.schemas.map import StationCollection, StationFeature, StationPoint, StationProperties

router = APIRouter(prefix="/map", tags=["map"])

STATIONS_TTL_SECONDS = 30


@router.get("/stations", response_model=StationCollection)
async def map_stations(
    request: Request,
    response: Response,
    repo: WeatherRepository = Depends(get_weather_repo),
    cache: TTLCache = Depends(get_cache),
) -> StationCollection | Response:
    """Latest observation per city, joined against station coordinates."""
    return await cached(
        request,
        response,
        cache,
        "map:stations",
        STATIONS_TTL_SECONDS,
        lambda: _collect(repo),
    )


async def _collect(repo: WeatherRepository) -> StationCollection:
    stations = await repo.latest_per_city()
    features: list[StationFeature] = []
    for station in stations:
        if station.latitude is None or station.longitude is None:
            continue
        features.append(
            StationFeature(
                geometry=StationPoint(coordinates=[station.longitude, station.latitude]),
                properties=StationProperties(
                    city=station.city,
                    temperature=station.temperature,
                    apparent_temperature=station.apparent_temperature,
                    relative_humidity=station.humidity,
                    wind_speed=station.wind_speed,
                    wind_direction=station.wind_direction,
                    precipitation=station.precipitation,
                    weather_code=station.weather_code,
                    observed_at=station.observed_at,
                ),
            )
        )
    return StationCollection(features=features)
