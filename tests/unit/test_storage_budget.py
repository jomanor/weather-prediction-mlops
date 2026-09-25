"""Steady-state Atlas M0 storage budget.

Atlas M0 is capped at 512 MiB. This module builds an explicit per-collection
steady-state model from *derived* inputs and measured document sizes:

* Retention windows are parsed from ``backend/app/db/mongo.py:TTL_INDEXES``
  (seconds → days). Nothing here restates them, so a TTL change flows straight
  into the arithmetic and can fail the budget.
* Station count is parsed from ``kafka/kafka-producer/producer.py:FALLBACK_CITIES``
  (the API seeds the same 14 stations).
* Bytes/row are **measured** from the production Atlas ``weather_db`` on
  2026-09-25 via read-only ``collStats.avgObjSize``:
  ``raw_weather=1362 B``, ``weather_features=3856 B``, ``weather_predictions=405 B``.
  ``weather_data`` was empty at measurement time, so it uses a documented
  estimate (600 B).

Rows are one observation per station per hour; ``weather_features`` is a single
overwritten snapshot bounded by the *raw* window (see the overwrite invariant
test), so it is charged ``raw_weather`` days, not its own TTL.

**The handoff's ~1.4 MB/day is treated as total-DB growth during the fill
phase**, not as a per-collection ingest rate: it is dominated by
``weather_features`` filling toward its 180-day snapshot. The model below does
not use it as an input; ``avgObjSize × rows`` is more precise. At steady state
both ``raw_weather`` (TTL) and ``weather_features`` (window) saturate, so net
growth falls to index churn plus the small ``weather_predictions``/``weather_data``
reservoirs.

Conclusion at the current constants: **≈334 MiB of 512 MiB**, under the 75%
(384 MiB) safety budget, with ``weather_features`` the largest single collection
(≈222 MiB, ~43% of the cap). If the raw window is extended (e.g. its TTL grows
past ~250 days), ``weather_features`` is the first axis to breach — the snapshot
scales linearly with the raw window and there is no TTL on the collection itself.
Update ``MEASURED_AVG_OBJ_SIZE`` when the schema or the measured sizes move.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.conftest import REPO_ROOT, eval_python_constant

# ---------------------------------------------------------------------------
# Sources (parsed, never imported: the modules need FastAPI/Motor/Spark)
# ---------------------------------------------------------------------------

MONGO_SOURCE = REPO_ROOT / "backend" / "app" / "db" / "mongo.py"
PRODUCER_SOURCE = REPO_ROOT / "kafka" / "kafka-producer" / "producer.py"
BATCH_PROCESSING_SOURCE = (
    Path(__file__).resolve().parents[2] / "spark" / "spark-jobs" / "batch_processing.py"
)

#: Derived from the installed TTL indexes: ``{collection: days}``.
RETENTION_DAYS: dict[str, int] = {
    collection: spec[0][1] // 86400
    for collection, spec in eval_python_constant(MONGO_SOURCE, "TTL_INDEXES").items()
}

#: Derived from the producer's station registry.
CITY_COUNT: int = len(eval_python_constant(PRODUCER_SOURCE, "FALLBACK_CITIES"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MB = 1024 * 1024
ATLAS_M0_CAP_MB = 512
ATLAS_M0_CAP_BYTES = ATLAS_M0_CAP_MB * MB
#: Fraction of the cap the model must stay under (25% headroom).
SAFETY_MARGIN = 0.75
HOURS_PER_DAY = 24

#: Measured ``collStats.avgObjSize`` on production ``weather_db``, 2026-09-25.
MEASURED_AVG_OBJ_SIZE: dict[str, int] = {
    "raw_weather": 1362,
    "weather_features": 3856,
    "weather_predictions": 405,
}
#: ``weather_data`` held 0 rows at measurement time; nested current-conditions
#: snapshot, estimated between the raw payload and a prediction document.
WEATHER_DATA_BYTES_PER_ROW_ESTIMATE = 600

#: Measured index size was ~2-6% of collection size per collection; budget 5%.
INDEX_OVERHEAD_FRACTION = 0.05


def rows_per_day() -> int:
    """One observation per station per hour across the registry."""
    return HOURS_PER_DAY * CITY_COUNT


def steady_state_bytes() -> dict[str, int]:
    """Per-collection steady-state bytes: retained rows × measured bytes/row."""
    per_day = rows_per_day()
    raw_days = RETENTION_DAYS["raw_weather"]
    rows = {
        # raw_weather accumulates for its full TTL window.
        "raw_weather": (raw_days * per_day, MEASURED_AVG_OBJ_SIZE["raw_weather"]),
        # weather_features is a single overwritten snapshot bounded by the raw
        # window (see test_weather_features_is_written_as_an_overwrite_snapshot).
        "weather_features": (raw_days * per_day, MEASURED_AVG_OBJ_SIZE["weather_features"]),
        "weather_data": (
            RETENTION_DAYS["weather_data"] * per_day,
            WEATHER_DATA_BYTES_PER_ROW_ESTIMATE,
        ),
        "weather_predictions": (
            RETENTION_DAYS["weather_predictions"] * per_day,
            MEASURED_AVG_OBJ_SIZE["weather_predictions"],
        ),
    }
    return {
        collection: row_count * bytes_per_row
        for collection, (row_count, bytes_per_row) in rows.items()
    }


def total_bytes() -> int:
    """All collections plus the index overhead."""
    return int(sum(steady_state_bytes().values()) * (1 + INDEX_OVERHEAD_FRACTION))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retention_days_track_mongo_ttl_indexes():
    """The windows must be the declared TTLs, not a literal copy."""
    declared = eval_python_constant(MONGO_SOURCE, "TTL_INDEXES")
    assert RETENTION_DAYS == {
        collection: spec[0][1] // 86400 for collection, spec in declared.items()
    }
    assert set(RETENTION_DAYS) == {"raw_weather", "weather_predictions", "weather_data"}
    assert all(days > 0 for days in RETENTION_DAYS.values())
    # The TTL fields themselves are part of the contract.
    assert declared["raw_weather"][0][0] == "timestamp"
    assert declared["weather_data"][0][0] == "timestamp"
    assert declared["weather_predictions"][0][0] == "prediction_timestamp"


def test_city_count_is_derived_from_the_producer_registry():
    assert CITY_COUNT == len(eval_python_constant(PRODUCER_SOURCE, "FALLBACK_CITIES"))
    assert CITY_COUNT > 0


def test_steady_state_is_under_cap_with_safety_margin():
    total = total_bytes()
    budget = ATLAS_M0_CAP_BYTES * SAFETY_MARGIN
    assert total < ATLAS_M0_CAP_BYTES, f"{total / MB:.1f} MiB exceeds the {ATLAS_M0_CAP_MB} MiB cap"
    assert (
        total < budget
    ), f"{total / MB:.1f} MiB exceeds the {SAFETY_MARGIN:.0%} budget {budget / MB:.1f} MiB"


def test_weather_features_is_the_largest_collection():
    """Names the first axis to breach if the raw window is extended."""
    sizes = steady_state_bytes()
    assert max(sizes, key=sizes.get) == "weather_features"
    assert sizes["weather_features"] > sizes["raw_weather"]


def test_safety_margin_is_meaningful():
    assert 0.0 < SAFETY_MARGIN < 1.0


# ---------------------------------------------------------------------------
# weather_features overwrite invariant — justifies treating it as a snapshot.
# ---------------------------------------------------------------------------


def _function_body(source: str, name: str) -> str:
    """Source of the top-level ``def name`` block (up to the next top-level def)."""
    start = source.index(f"def {name}(")
    tail = source[start:]
    following = re.search(r"\ndef ", tail[1:])
    return tail if following is None else tail[: following.start() + 1]


def test_weather_features_is_written_as_an_overwrite_snapshot():
    body = _function_body(BATCH_PROCESSING_SOURCE.read_text(), "save_features_to_mongodb")
    compact = re.sub(r"\s+", "", body)

    # The function defaults to the weather_features collection ...
    assert 'collection_name="weather_features"' in compact
    # ... and writes it with overwrite, not append. The repo's actual write
    # options build as: .option("collection", collection_name).mode("overwrite")
    assert '.option("collection",collection_name).mode("overwrite")' in compact
    assert 'mode("append")' not in compact
