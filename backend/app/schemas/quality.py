"""Data-quality schemas (``docs/api-contract.md`` → ``GET /api/weather/quality``).

Per-city view of the model's actual input collection (``weather_features``):
completeness against the expected hourly grid, the largest gap, the null rate
and the age of the last observation. Nullable fields are always present.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

QualityStatus = Literal["ok", "warn", "bad"]


class CityQuality(BaseModel):
    """One city's data-quality meter over the requested window."""

    city: str
    expected_hours: int
    observed_hours: int
    completeness: float
    max_gap_hours: float | None = None
    null_rate: float | None = None
    last_observed_at: datetime | None = None
    age_hours: float | None = None
    status: QualityStatus


class WeatherQuality(BaseModel):
    generated_at: datetime
    days: int
    cities: list[CityQuality]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "generated_at": "2026-09-25T12:00:00Z",
                "days": 7,
                "cities": [
                    {
                        "city": "Madrid",
                        "expected_hours": 168,
                        "observed_hours": 165,
                        "completeness": 0.982,
                        "max_gap_hours": 3.0,
                        "null_rate": 0.012,
                        "last_observed_at": "2026-09-25T11:00:00Z",
                        "age_hours": 1.0,
                        "status": "ok",
                    }
                ],
            }
        }
    )
