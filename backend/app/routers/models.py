"""Model registry endpoint: ``/api/models``."""

from fastapi import APIRouter, Depends, Request, Response

from app.core.cache import TTLCache, cached, get_cache
from app.repositories.model_repo import ModelRepository, get_model_repo
from app.schemas.models import ModelsResponse

router = APIRouter(prefix="/models", tags=["models"])

MODELS_TTL_SECONDS = 300


@router.get("", response_model=ModelsResponse)
async def list_models(
    request: Request,
    response: Response,
    repo: ModelRepository = Depends(get_model_repo),
    cache: TTLCache = Depends(get_cache),
) -> ModelsResponse | Response:
    async def load() -> ModelsResponse:
        models = await repo.list_models()
        return ModelsResponse(count=len(models), models=models)

    return await cached(request, response, cache, "models:list", MODELS_TTL_SECONDS, load)
