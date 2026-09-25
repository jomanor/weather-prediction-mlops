"""Prediction endpoints: ``/api/predictions/*``."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.repositories.prediction_repo import PredictionRepository, get_prediction_repo
from app.schemas.predictions import LatestPredictionsResponse, Prediction

router = APIRouter(prefix="/predictions", tags=["predictions"])

#: Client-supplied horizon bound; the M2 config (WS-A) emits [1, 3, 6, 12, 24].
HORIZON_QUERY = Query(default=None, ge=1, le=48, description="Filter by forecast horizon (hours)")


@router.get("/latest", response_model=LatestPredictionsResponse)
async def latest_predictions(
    horizon: int | None = HORIZON_QUERY,
    repo: PredictionRepository = Depends(get_prediction_repo),
) -> LatestPredictionsResponse:
    # Omitted -> one row per (city, horizon); provided -> one row per city.
    predictions = await repo.latest_per_city(horizon=horizon)
    generated_at = max((p.prediction_timestamp for p in predictions), default=None)
    return LatestPredictionsResponse(
        count=len(predictions), generated_at=generated_at, predictions=predictions
    )


@router.get("/{city}", response_model=list[Prediction])
async def predictions_for_city(
    city: str,
    limit: int = Query(default=48, ge=1, le=500, description="Maximum number of records"),
    horizon: int | None = HORIZON_QUERY,
    repo: PredictionRepository = Depends(get_prediction_repo),
) -> list[Prediction]:
    predictions = await repo.for_city(city, limit=limit, horizon=horizon)
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for city '{city}'. Run the inference job first.",
        )
    # ``for_city`` returns newest-first so ``limit`` keeps the latest window;
    # serve chronological order because charts read oldest to newest.
    return list(reversed(predictions))
