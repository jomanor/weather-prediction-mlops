"""Analytics schemas (``docs/batch4-contract.md`` → Contract 4).

Every field is null-tolerant: a missing observation serializes as ``null``
instead of a fabricated ``0``. Counts (``n``/``count``) default to ``0``;
nothing else is invented.
"""

from datetime import datetime

from pydantic import BaseModel


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
