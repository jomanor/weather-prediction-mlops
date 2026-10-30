"""Model prediction schemas (``docs/api-contract.md`` → Prediction)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.core.coerce import as_utc


class Prediction(BaseModel):
    city: str
    source_timestamp: datetime
    prediction_timestamp: datetime
    horizon_hours: int
    predicted_temperature: float | None = None
    predicted_rain: float | None = None
    observed_temperature: float | None = None
    temp_lower: float | None = None
    temp_upper: float | None = None
    interval_level: float | None = None
    temp_model_name: str | None = None
    temp_model_version: str | None = None
    rain_model_name: str | None = None
    rain_model_version: str | None = None
    #: Age of ``source_timestamp`` at serialization time. ``None`` when the
    #: timestamp is missing (never invent an age); ``stale`` then stays False.
    source_age_hours: float | None = None
    stale: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Madrid",
                "source_timestamp": "2026-09-23T13:00:00Z",
                "prediction_timestamp": "2026-09-23T14:00:00Z",
                "horizon_hours": 1,
                "predicted_temperature": 23.6,
                "predicted_rain": 0.0,
                "observed_temperature": 24.1,
                "temp_lower": 21.7,
                "temp_upper": 25.7,
                "interval_level": 0.8,
                "temp_model_name": "temp_prediction_1h_GradientBoostedTrees",
                "temp_model_version": "20260923_020000",
                "rain_model_name": "rain_prediction_1h_GradientBoostedTrees",
                "rain_model_version": "20260923_020000",
                "source_age_hours": 1.0,
                "stale": False,
            }
        }
    )


def freshness_fields(
    source_timestamp: datetime | None, now: datetime, stale_after_hours: float
) -> dict[str, Any]:
    """``source_age_hours``/``stale`` for a prediction, computed at serialization.

    A missing (or unparseable) ``source_timestamp`` yields ``None``/``False``:
    the API must never invent an age for a row it cannot date.
    """
    timestamp = as_utc(source_timestamp)
    if timestamp is None:
        return {"source_age_hours": None, "stale": False}
    age_hours = round((now - timestamp).total_seconds() / 3600, 1)
    return {"source_age_hours": age_hours, "stale": age_hours > stale_after_hours}


class LatestPredictionsResponse(BaseModel):
    count: int
    generated_at: datetime | None = None
    predictions: list[Prediction]
