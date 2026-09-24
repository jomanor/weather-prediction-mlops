"""Model prediction schemas (``docs/api-contract.md`` → Prediction)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Prediction(BaseModel):
    city: str
    source_timestamp: datetime
    prediction_timestamp: datetime
    horizon_hours: int
    predicted_temperature: float | None = None
    predicted_rain: float | None = None
    observed_temperature: float | None = None
    temp_model_name: str | None = None
    temp_model_version: str | None = None
    rain_model_name: str | None = None
    rain_model_version: str | None = None

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
                "temp_model_name": "temp_prediction_1h_GradientBoostedTrees",
                "temp_model_version": "20260923_020000",
                "rain_model_name": "rain_prediction_1h_GradientBoostedTrees",
                "rain_model_version": "20260923_020000",
            }
        }
    )


class LatestPredictionsResponse(BaseModel):
    count: int
    generated_at: datetime | None = None
    predictions: list[Prediction]
