"""Benchmark schemas: model vs AEMET vs observed (``docs/api-contract.md``)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Metrics(BaseModel):
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None
    n: int = 0

    model_config = ConfigDict(protected_namespaces=())


class SeriesPoint(BaseModel):
    timestamp: datetime
    observed: float | None = None
    model: float | None = None
    aemet: float | None = None
    residual_model: float | None = None
    residual_aemet: float | None = None

    model_config = ConfigDict(protected_namespaces=())


class AemetStatus(BaseModel):
    available: bool
    error: str | None = None
    issued_at: datetime | None = None


class MetricsPair(BaseModel):
    model: Metrics
    aemet: Metrics

    model_config = ConfigDict(protected_namespaces=())


class BenchmarkResponse(BaseModel):
    city: str
    hours: int
    generated_at: datetime
    aemet: AemetStatus
    series: list[SeriesPoint]
    metrics: MetricsPair


class CityBenchmark(BaseModel):
    city: str
    n: int
    model: Metrics
    aemet: Metrics

    model_config = ConfigDict(protected_namespaces=())


class BenchmarkSummary(BaseModel):
    generated_at: datetime
    aemet_configured: bool
    cities: list[CityBenchmark]
