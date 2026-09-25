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
    # Honest-metrics block (Batch 3, Contract 3). ``None`` when the registry
    # document predates the temporal split / baseline work — never invented.
    persistence_rmse: float | None = None
    climatology_rmse: float | None = None
    skill_score: float | None = None
    brier: float | None = None
    persistence_brier: float | None = None
    prevalence: float | None = None
    coverage: float | None = None


class ModelInfo(BaseModel):
    name: str
    version: str | None = None
    target: str | None = None
    horizon_hours: int | None = None
    created_at: datetime | None = None
    metrics: ModelMetrics | None = None
    stage: str | None = None
    split: dict | None = None
    interval: dict | None = None
    commit: str | None = None

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
                "split": {
                    "kind": "temporal",
                    "train_end": "2026-09-16T00:00:00Z",
                    "val_end": "2026-09-19T00:00:00Z",
                    "test_start": "2026-09-19T00:00:00Z",
                },
                "interval": {"level": 0.8, "lower_offset": -1.9, "upper_offset": 2.1},
                "commit": 'GITHUB_SHA or "unknown"',
            }
        },
    )


class ModelsResponse(BaseModel):
    count: int
    models: list[ModelInfo]
