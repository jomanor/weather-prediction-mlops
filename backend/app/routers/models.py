"""Model registry endpoint: ``/api/models``."""

from fastapi import APIRouter, Depends

from app.repositories.model_repo import ModelRepository, get_model_repo
from app.schemas.models import ModelsResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsResponse)
async def list_models(repo: ModelRepository = Depends(get_model_repo)) -> ModelsResponse:
    models = await repo.list_models()
    return ModelsResponse(count=len(models), models=models)
