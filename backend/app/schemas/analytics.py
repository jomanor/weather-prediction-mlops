"""Analytics schemas (``docs/batch4-contract.md`` → Contract 4).

Every field is null-tolerant: a missing observation serializes as ``null``
instead of a fabricated ``0``. Counts (``n``/``count``) default to ``0`` and the
``heatwave`` flag defaults to ``False``; nothing else is invented.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DailyPoint(BaseModel):
    """One calendar day of aggregates for a city."""

    date: str  # YYYY-MM-DD (UTC)
    tmin: float | None = None  # °C
    tmax: float | None = None  # °C
    tmean: float | None = None  # °C
    hdd: float | None = None  # heating degree days, base 18 °C
    cdd: float | None = None  # cooling degree days, base 18 °C
    anomaly: float | None = None  # tmean − same-city day-of-year climatology mean
    heatwave: bool = False  # part of a >= 3 consecutive day tmax >= 35 °C run

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "date": "2026-09-24",
                "tmin": 14.2,
                "tmax": 31.8,
                "tmean": 23.1,
                "hdd": 0.0,
                "cdd": 5.1,
                "anomaly": 1.4,
                "heatwave": False,
            }
        }
    )


class DailyResponse(BaseModel):
    city: str
    days: int
    generated_at: datetime
    points: list[DailyPoint]


class ClimatologyPoint(BaseModel):
    day_of_year: int  # 1..366
    tmean: float | None = None
    tmin: float | None = None
    tmax: float | None = None
    n: int = 0


class ClimatologyResponse(BaseModel):
    city: str
    generated_at: datetime
    basis_years: float  # distinct calendar years the series is built from
    series: list[ClimatologyPoint]


class WindSector(BaseModel):
    sector: int  # 0..15, clockwise from N
    count: int = 0
    mean_speed: float | None = None  # km/h


class WindRoseResponse(BaseModel):
    city: str
    days: int
    generated_at: datetime
    sectors: list[WindSector]


class DiurnalCell(BaseModel):
    city: str
    hour: int  # local hour 0..23
    tmean: float | None = None
    n: int = 0


class DiurnalResponse(BaseModel):
    days: int
    generated_at: datetime
    cells: list[DiurnalCell]


class CorrelationResponse(BaseModel):
    days: int
    var: str
    generated_at: datetime
    cities: list[str]
    matrix: list[list[float | None]]


class ErrorByHourPoint(BaseModel):
    horizon_hours: int
    hour: int  # local hour of the verified observation
    n: int = 0
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None
    persistence_mae: float | None = None


class ErrorByHourResponse(BaseModel):
    days: int
    generated_at: datetime
    points: list[ErrorByHourPoint]
