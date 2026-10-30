"""Refresh endpoint schemas (``POST /api/refresh``)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.routers.cities import MAX_NAME_LENGTH


class RefreshRequest(BaseModel):
    city: str = Field(..., min_length=1, max_length=MAX_NAME_LENGTH)


class RefreshResponse(BaseModel):
    """Honest result of one refresh attempt.

    Both booleans are always false today: the endpoint only writes the raw
    observation. Feature engineering and inference run as scheduled Spark jobs,
    never from a request, so they are reported rather than implied.
    """

    city: str
    source_timestamp: datetime
    fetched_at: datetime
    features_refreshed: bool = False
    predictions_refreshed: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Madrid",
                "source_timestamp": "2026-09-23T12:00:00Z",
                "fetched_at": "2026-09-23T12:00:05Z",
                "features_refreshed": False,
                "predictions_refreshed": False,
            }
        }
    )
