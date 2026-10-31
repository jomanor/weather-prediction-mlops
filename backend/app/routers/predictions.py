"""Prediction endpoints: ``/api/predictions/*``."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.core.cache import TTLCache, cached, get_cache
from app.core.thresholds import PREDICTION_STALE_AFTER_HOURS
from app.repositories.prediction_repo import PredictionRepository, get_prediction_repo
from app.schemas.predictions import LatestPredictionsResponse, Prediction, freshness_fields

router = APIRouter(prefix="/predictions", tags=["predictions"])

#: Client-supplied horizon bound; the M2 config (WS-A) emits [1, 3, 6, 12, 24].
HORIZON_QUERY = Query(default=None, ge=1, le=48, description="Filter by forecast horizon (hours)")

#: ``/latest`` runs an aggregate over ``weather_predictions`` per request, so it
#: is cached for the same short window as the other hot reads. Age and ``stale``
#: are stamped inside the loader against one ``now``, so a hit can only lag the
#: clock by the TTL — far below the 12 h staleness threshold.
LATEST_TTL_SECONDS = 30


def _with_freshness(predictions: list[Prediction]) -> list[Prediction]:
    """Stamp ``source_age_hours``/``stale`` on every row against one ``now``."""
    now = datetime.now(timezone.utc)
    return [
        prediction.model_copy(
            update=freshness_fields(prediction.source_timestamp, now, PREDICTION_STALE_AFTER_HOURS)
        )
        for prediction in predictions
    ]


def _latest_key(horizon: int | None) -> str:
    """Cache key: ``horizon`` is the only query parameter that changes the payload."""
    return f"predictions:latest:{horizon if horizon is not None else 'all'}"


@router.get("/latest", response_model=LatestPredictionsResponse)
async def latest_predictions(
    request: Request,
    response: Response,
    horizon: int | None = HORIZON_QUERY,
    repo: PredictionRepository = Depends(get_prediction_repo),
    cache: TTLCache = Depends(get_cache),
) -> LatestPredictionsResponse | Response:
    async def load() -> LatestPredictionsResponse:
        # Omitted -> one row per (city, horizon); provided -> one row per city.
        predictions = await repo.latest_per_city(horizon=horizon)
        generated_at = max((p.prediction_timestamp for p in predictions), default=None)
        return LatestPredictionsResponse(
            count=len(predictions),
            generated_at=generated_at,
            predictions=_with_freshness(predictions),
        )

    return await cached(request, response, cache, _latest_key(horizon), LATEST_TTL_SECONDS, load)


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
    return _with_freshness(list(reversed(predictions)))
