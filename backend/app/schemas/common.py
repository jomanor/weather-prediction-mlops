"""Types shared by several endpoints. Mirrors ``docs/api-contract.md``."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    time: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "service": "weather-api",
                "version": "2.0.0",
                "time": "2026-09-23T14:00:00Z",
            }
        }
    )


class ErrorResponse(BaseModel):
    detail: str
