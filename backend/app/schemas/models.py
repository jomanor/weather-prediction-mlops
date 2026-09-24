"""Model registry schemas (``docs/api-contract.md`` → GET /api/models).

``model_registry`` documents are written by ``spark/spark-jobs/ml_training.py``
with ``model_name``, ``model_type``, ``gridfs_file_id``, ``timestamp`` and
``version``. ``target`` and ``horizon_hours`` are derived from ``model_name``;
``metrics`` and ``stage`` are passed through when the registry document has
them (MLflow-registered models do), otherwise they are null — never invented.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelMetrics(BaseModel):
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None


class ModelInfo(BaseModel):
    name: str
    version: str | None = None
    target: str | None = None
    horizon_hours: int | None = None
    created_at: datetime | None = None
    metrics: ModelMetrics | None = None
    stage: str | None = None

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "name": "temp_prediction_1h_GradientBoostedTrees",
                "version": "20260923_020000",
                "target": "temperature",
                "horizon_hours": 1,
                "created_at": "2026-09-23T02:00:00Z",
                "metrics": {"rmse": 1.44, "mae": 1.12, "r2": 0.91},
                "stage": "production",
            }
        },
    )


class ModelsResponse(BaseModel):
    count: int
    models: list[ModelInfo]
