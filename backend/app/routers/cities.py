"""Station registry (``/api/cities``) and geocoding proxy (``/api/geo/search``)."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.core.cache import TTLCache, cached, get_cache
from app.repositories.city_repo import CityRepository, get_city_repo
from app.schemas.city import City, GeoSearchResponse
from app.services.geo import GeocodingError, GeocodingService, get_geo_service

router = APIRouter(tags=["cities"])

MAX_NAME_LENGTH = 100
CITIES_TTL_SECONDS = 60
GEO_TTL_SECONDS = 300


@router.get("/cities", response_model=list[City])
async def list_cities(
    request: Request,
    response: Response,
    repo: CityRepository = Depends(get_city_repo),
    cache: TTLCache = Depends(get_cache),
) -> list[City] | Response:
    return await cached(request, response, cache, "cities:list", CITIES_TTL_SECONDS, repo.list)


@router.post("/cities", response_model=City, status_code=201)
async def create_city(city: City, repo: CityRepository = Depends(get_city_repo)) -> City:
    name = city.name.strip()
    if not name or len(name) > MAX_NAME_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"name must be 1 to {MAX_NAME_LENGTH} characters"
        )
    if not -90.0 <= city.latitude <= 90.0:
        raise HTTPException(status_code=400, detail="latitude must be between -90 and 90")
    if not -180.0 <= city.longitude <= 180.0:
        raise HTTPException(status_code=400, detail="longitude must be between -180 and 180")

    if await repo.get(name) is not None:
        raise HTTPException(status_code=409, detail=f"City '{name}' already exists")

    return await repo.upsert(City(name=name, latitude=city.latitude, longitude=city.longitude))


@router.delete("/cities/{name}", status_code=204)
async def delete_city(name: str, repo: CityRepository = Depends(get_city_repo)) -> None:
    if not await repo.delete(name):
        raise HTTPException(status_code=404, detail=f"City '{name}' not found")


@router.get("/geo/search", response_model=GeoSearchResponse)
async def geo_search(
    request: Request,
    response: Response,
    q: str | None = Query(default=None, description="Place name to geocode"),
    service: GeocodingService = Depends(get_geo_service),
    cache: TTLCache = Depends(get_cache),
) -> GeoSearchResponse | Response:
    query = q or ""

    async def load() -> GeoSearchResponse:
        try:
            results = await service.search(query)
        except GeocodingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return GeoSearchResponse(results=results)

    return await cached(
        request,
        response,
        cache,
        f"geo:{query.strip().lower()}",
        GEO_TTL_SECONDS,
        load,
    )
