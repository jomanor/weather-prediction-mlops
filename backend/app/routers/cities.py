"""Station registry (``/api/cities``) and geocoding proxy (``/api/geo/search``).

The two registry writes are gated (see :func:`require_city_writes_token`): the
registry is the source of truth for ingestion, so deleting a station silently
stops its ingestion and degrades the quality meter. Reads stay public.
"""

import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.core.cache import TTLCache, cached, get_cache
from app.core.config import Settings
from app.core.errors import APIError
from app.repositories.city_repo import CityRepository, get_city_repo
from app.schemas.city import City, GeoSearchResponse
from app.services.geo import GeocodingError, GeocodingService, get_geo_service

router = APIRouter(tags=["cities"])

#: Cached ``GET /cities`` payload. Named once because the writes below must
#: invalidate exactly this key.
CITIES_CACHE_KEY = "cities:list"

MAX_NAME_LENGTH = 100
CITIES_TTL_SECONDS = 60
GEO_TTL_SECONDS = 300

WRITES_DISABLED_MESSAGE = "City writes are disabled"
UNAUTHORIZED_MESSAGE = "Missing or invalid city writes token"
#: Sent on every 401 so a browser client knows the expected scheme.
BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}


def _presented_token(authorization: str | None) -> str | None:
    """The credentials of an ``Authorization: Bearer <token>`` header, or ``None``."""
    if not authorization:
        return None
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    credentials = credentials.strip()
    return credentials or None


def require_city_writes_token(request: Request) -> None:
    """Gate a registry write on the configured shared secret.

    Unset token => 503 (writes disabled, reads untouched); set => an
    ``Authorization: Bearer <token>`` header must match it. The comparison is
    constant time and neither the configured nor the submitted token is ever
    echoed in the body or written to a log (the access line carries no headers).
    """
    settings: Settings = request.app.state.settings
    token = settings.city_writes_token
    if token is None:
        raise APIError(503, "city_writes_disabled", WRITES_DISABLED_MESSAGE)

    presented = _presented_token(request.headers.get("authorization"))
    if presented is None or not secrets.compare_digest(
        presented.encode("utf-8"), token.encode("utf-8")
    ):
        raise APIError(401, "unauthorized", UNAUTHORIZED_MESSAGE, headers=BEARER_CHALLENGE)


@router.get("/cities", response_model=list[City])
async def list_cities(
    request: Request,
    response: Response,
    repo: CityRepository = Depends(get_city_repo),
    cache: TTLCache = Depends(get_cache),
) -> list[City] | Response:
    return await cached(request, response, cache, CITIES_CACHE_KEY, CITIES_TTL_SECONDS, repo.list)


@router.post(
    "/cities",
    response_model=City,
    status_code=201,
    dependencies=[Depends(require_city_writes_token)],
)
async def create_city(
    city: City,
    repo: CityRepository = Depends(get_city_repo),
    cache: TTLCache = Depends(get_cache),
) -> City:
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

    created = await repo.upsert(City(name=name, latitude=city.latitude, longitude=city.longitude))
    # Only after the write landed: a 400/401/409/503 must leave the cached list
    # (and its in-flight load) alone, or it would be dropped for nothing.
    cache.delete(CITIES_CACHE_KEY)
    return created


@router.delete(
    "/cities/{name}",
    status_code=204,
    dependencies=[Depends(require_city_writes_token)],
)
async def delete_city(
    name: str,
    repo: CityRepository = Depends(get_city_repo),
    cache: TTLCache = Depends(get_cache),
) -> None:
    if not await repo.delete(name):
        raise HTTPException(status_code=404, detail=f"City '{name}' not found")
    # Invalidate after the delete, never before: a 404 leaves the list untouched.
    cache.delete(CITIES_CACHE_KEY)


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
