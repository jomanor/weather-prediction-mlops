"""Coercion helpers shared by the repository mappers.

MongoDB documents come from three writers (Kafka consumer, backfill script,
Spark jobs) and are not fully consistent: numbers may be strings, timestamps
may be naive or aware, optional keys may be missing or empty. These helpers
turn any of those into the strict types the API contract promises.
"""

from datetime import datetime, timezone
from typing import Any


def as_utc(value: Any) -> datetime | None:
    """Return *value* as a timezone-aware UTC datetime, or None."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        # Mongo stores naive datetimes as UTC; treat them as such.
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_float(value: Any) -> float | None:
    """Best-effort float, None for missing/blank/non-numeric values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    return None if number is None else int(number)


def first_not_none(*values: Any) -> Any:
    """First value that is neither None nor an empty string."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None
