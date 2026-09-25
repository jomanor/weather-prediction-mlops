"""MongoDB access for ``model_registry`` (GridFS/MLflow model metadata)."""

import re
from typing import Any

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.coerce import as_utc, to_float, to_int
from app.db.mongo import get_db
from app.schemas.models import (
    ModelDiagnostics,
    ModelInfo,
    ModelMetrics,
    SliceMetric,
)

# e.g. temp_prediction_1h_GradientBoostedTrees / rain_prediction_3h_RandomForest
_NAME_RE = re.compile(r"^(?P<kind>temp|rain)_prediction_(?P<horizon>\d+)h", re.IGNORECASE)

_TARGETS = {"temp": "temperature", "rain": "rain"}


def _target_from_name(name: str) -> str | None:
    match = _NAME_RE.match(name)
    return _TARGETS.get(match.group("kind").lower()) if match else None


def _horizon_from_name(name: str) -> int | None:
    match = _NAME_RE.match(name)
    return int(match.group("horizon")) if match else None


def _slice_metrics(raw: Any) -> list[SliceMetric]:
    """Map a registry list of slice dicts; a malformed entry is dropped."""
    if not isinstance(raw, list):
        return []
    slices: list[SliceMetric] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str):
            continue
        slices.append(
            SliceMetric(
                label=label,
                n=to_int(item.get("n")) or 0,
                rmse=to_float(item.get("rmse")),
                mae=to_float(item.get("mae")),
                bias=to_float(item.get("bias")),
                brier=to_float(item.get("brier")),
            )
        )
    return slices


def _diagnostics_from_doc(raw: Any) -> ModelDiagnostics | None:
    """Map the additive Batch 4 ``diagnostics`` block; legacy docs -> None."""
    if not isinstance(raw, dict):
        return None
    drift = raw.get("drift_psi")
    return ModelDiagnostics(
        by_city=_slice_metrics(raw.get("by_city")),
        by_hour_of_day=_slice_metrics(raw.get("by_hour_of_day")),
        by_rain_bucket=_slice_metrics(raw.get("by_rain_bucket")),
        drift_psi=(
            {key: to_float(value) for key, value in drift.items()}
            if isinstance(drift, dict)
            else {}
        ),
    )


def model_from_doc(doc: dict[str, Any]) -> ModelInfo | None:
    """``model_registry`` document -> ModelInfo."""
    name = doc.get("model_name") or doc.get("name")
    if not name:
        return None

    raw_metrics = doc.get("metrics")
    metrics = None
    if isinstance(raw_metrics, dict):
        metrics = ModelMetrics(
            rmse=to_float(raw_metrics.get("rmse")),
            mae=to_float(raw_metrics.get("mae")),
            r2=to_float(raw_metrics.get("r2")),
            persistence_rmse=to_float(raw_metrics.get("persistence_rmse")),
            climatology_rmse=to_float(raw_metrics.get("climatology_rmse")),
            skill_score=to_float(raw_metrics.get("skill_score")),
            brier=to_float(raw_metrics.get("brier")),
            persistence_brier=to_float(raw_metrics.get("persistence_brier")),
            prevalence=to_float(raw_metrics.get("prevalence")),
            coverage=to_float(raw_metrics.get("coverage")),
        )

    split = doc.get("split")
    interval = doc.get("interval")
    commit = doc.get("commit")
    return ModelInfo(
        name=name,
        version=doc.get("version"),
        target=doc.get("target") or _target_from_name(name),
        horizon_hours=to_int(doc.get("horizon_hours")) or _horizon_from_name(name),
        created_at=as_utc(doc.get("timestamp")),
        metrics=metrics,
        stage=doc.get("stage") or doc.get("current_stage"),
        split=split if isinstance(split, dict) else None,
        interval=interval if isinstance(interval, dict) else None,
        commit=commit if isinstance(commit, str) else None,
        diagnostics=_diagnostics_from_doc(doc.get("diagnostics")),
    )


class ModelRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    @property
    def _collection(self):
        return self._db["model_registry"]

    async def list_models(self, limit: int = 100) -> list[ModelInfo]:
        # ``model_registry`` has a ``{model_name: 1, timestamp: -1}`` index and no
        # city field, so this is the index-compatible sort (city-prefix rule N/A).
        cursor = self._collection.find({}).sort([("model_name", 1), ("timestamp", -1)]).limit(limit)
        docs = [doc async for doc in cursor]
        return [model for model in (model_from_doc(doc) for doc in docs) if model is not None]


def get_model_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> ModelRepository:
    return ModelRepository(db)
