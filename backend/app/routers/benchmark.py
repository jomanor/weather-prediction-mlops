"""Benchmark endpoints: ``/api/benchmark`` and ``/api/benchmark/{city}``."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.benchmark import BenchmarkResponse, BenchmarkSummary
from app.services.benchmark import BenchmarkService, CityDataNotFound, get_benchmark_service

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

HOURS_QUERY = Query(default=24, ge=1, le=168, description="Hours to compare (max 168)")


@router.get("", response_model=BenchmarkSummary)
async def benchmark_summary(
    hours: int = HOURS_QUERY,
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkSummary:
    return await service.summary(hours=hours)


@router.get("/{city}", response_model=BenchmarkResponse)
async def benchmark_for_city(
    city: str,
    hours: int = HOURS_QUERY,
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResponse:
    try:
        return await service.city_benchmark(city, hours=hours)
    except CityDataNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"No observations or predictions found for city '{city}'",
        ) from None
