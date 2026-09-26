"""Steady-state Atlas M0 storage budget.

Atlas M0 is capped at 512 MiB. This module builds an explicit per-collection
steady-state model from *derived* inputs and measured document sizes:

* Retention windows are parsed from ``backend/app/db/mongo.py:TTL_INDEXES``
  (seconds → days). Nothing here restates them, so a TTL change flows straight
  into the arithmetic and can fail the budget.
* Horizons are parsed from ``spark/config/spark_config.py:FEATURES_CONFIG``
  (M2 ``target_horizons``; ``int | list`` is accepted). If the key is not there
  yet the frozen M2 set is used, so the model never silently reverts to the
  1-horizon volume.
* The *live* station count is **measured** (15, incl. Pasadena). The producer's
  ``FALLBACK_CITIES`` still lists 14 — a known registry gap (separate follow-up,
  not Batch 4) — so it is only checked as a lower bound, never used as the
  budget input.
* Bytes/row are **measured** from the production Atlas ``weather_db`` on
  2026-09-25 via read-only ``collStats.avgObjSize``:
  ``raw_weather=1362 B``, ``weather_features=3856 B``,
  ``weather_predictions=437 B`` (Batch-3 interval columns grew it from 405 B).
  ``weather_data`` was empty at measurement time, so it uses a documented
  estimate (600 B).

Rows are one observation per station per hour; ``weather_features`` is a single
overwritten snapshot bounded by the *raw* window (see the overwrite invariant
test), so it is charged ``raw_weather`` days, not its own TTL.

**M2 (Batch 4):** inference scores every configured horizon per run and writes
one row per ``(city, source_timestamp, horizon)`` with ``operationType=replace``
(upsert). A run therefore adds rows **only when the latest feature
``source_timestamp`` advances** — the binding rate is the feature-rebuild
cadence, *not* the hourly inference schedule (which would be physically
impossible with the upsert and reports a false fail). Predictions rows/day are
``FEATURE_REBUILDS_PER_DAY × cities × horizons``, with
``FEATURE_REBUILDS_PER_DAY = 8``: 2× the declared ``20 */6`` = 4/day, as cadence
jitter headroom.

**The handoff's ~1.4 MB/day is treated as total-DB growth during the fill
phase**, not as a per-collection ingest rate: it is dominated by
``weather_features`` filling toward its 180-day snapshot. The model below does
not use it as an input; ``avgObjSize × rows`` is more precise. At steady state
both ``raw_weather`` (TTL) and ``weather_features`` (window) saturate, so net
growth falls to index churn plus the small ``weather_predictions``/``weather_data``
reservoirs.

Conclusion at the current constants: **≈368.7 MiB of 512 MiB**, under the 75%
(384 MiB) safety budget with ≈15 MiB headroom, and ``weather_features`` the
largest single collection (≈238 MiB, ~47% of the cap). The tripwire is the
feature-rebuild rate: **the first breaching rate is 14 rebuilds/day** (13/day
projects ≈383.5 MiB, just under the budget; 14/day ≈386.4 MiB, over — the
contract states the same 14/day boundary). Any cadence change (e.g. the deferred L1
15-min ingest) requires a re-measure and re-triggers this check. If the raw
window is extended (e.g. its TTL grows past ~250 days), ``weather_features`` is
the first axis to breach — the snapshot scales linearly with the raw window and
there is no TTL on the collection itself. Update ``MEASURED_AVG_OBJ_SIZE`` and
``LIVE_CITY_COUNT`` when the schema or the live data move.
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
SPARK_CONFIG_SOURCE = REPO_ROOT / "spark" / "config" / "spark_config.py"
BATCH_PROCESSING_SOURCE = (
    Path(__file__).resolve().parents[2] / "spark" / "spark-jobs" / "batch_processing.py"
)

#: Derived from the installed TTL indexes: ``{collection: days}``.
RETENTION_DAYS: dict[str, int] = {
    collection: spec[0][1] // 86400
    for collection, spec in eval_python_constant(MONGO_SOURCE, "TTL_INDEXES").items()
}

#: Producer registry — the live count is higher (Pasadena is missing there).
PRODUCER_CITY_COUNT: int = len(eval_python_constant(PRODUCER_SOURCE, "FALLBACK_CITIES"))

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
    "weather_predictions": 437,
}
#: ``weather_data`` held 0 rows at measurement time; nested current-conditions
#: snapshot, estimated between the raw payload and a prediction document.
WEATHER_DATA_BYTES_PER_ROW_ESTIMATE = 600

#: Measured live station count (includes Pasadena, absent from FALLBACK_CITIES).
LIVE_CITY_COUNT = 15

#: M2 ``target_horizons`` if ``spark_config`` predates the WS-A change.
FROZEN_M2_HORIZONS: tuple[int, ...] = (1, 3, 6, 12, 24)
#: Feature-rebuild cadence: 2× the declared ``20 */6`` = 4/day, for jitter.
FEATURE_REBUILDS_PER_DAY = 8

#: Measured index size was ~2-6% of collection size per collection; budget 5%.
INDEX_OVERHEAD_FRACTION = 0.05


def _normalize_horizons(value) -> tuple[int, ...]:
    """Accept the M2 ``list`` or a single ``int`` horizon."""
    if isinstance(value, bool):
        raise TypeError("horizons must be int or list, not bool")
    if isinstance(value, int):
        return (value,)
    return tuple(int(horizon) for horizon in value)


def configured_horizons() -> tuple[int, ...]:
    """``FEATURES_CONFIG`` horizons; frozen M2 set until WS-A lands the key."""
    config = eval_python_constant(SPARK_CONFIG_SOURCE, "FEATURES_CONFIG")
    if "target_horizons" in config:
        return _normalize_horizons(config["target_horizons"])
    return FROZEN_M2_HORIZONS


HORIZONS: tuple[int, ...] = configured_horizons()


def rows_per_day() -> int:
    """One observation per live station per hour across the registry."""
    return HOURS_PER_DAY * LIVE_CITY_COUNT


def prediction_rows_per_day(rebuilds: int | None = None, horizons: int | None = None) -> int:
    """Rows added per day: feature rebuilds × cities × horizons.

    Bounded by the feature-rebuild cadence, not the hourly inference cron: the
    ``operationType=replace`` upsert reuses each ``(city, source_timestamp,
    horizon)`` ``_id``, so hourly runs without a new ``source_timestamp`` add
    nothing.
    """
    rebuild_rate = FEATURE_REBUILDS_PER_DAY if rebuilds is None else rebuilds
    horizon_count = len(HORIZONS) if horizons is None else horizons
    return rebuild_rate * LIVE_CITY_COUNT * horizon_count


def steady_state_bytes(rebuilds: int | None = None, horizons: int | None = None) -> dict[str, int]:
    """Per-collection steady-state bytes: retained rows × measured bytes/row.

    *rebuilds*/*horizons* override the constants for sensitivity checks.
    """
    per_day = rows_per_day()
    predictions_per_day = prediction_rows_per_day(rebuilds, horizons)
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
            RETENTION_DAYS["weather_predictions"] * predictions_per_day,
            MEASURED_AVG_OBJ_SIZE["weather_predictions"],
        ),
    }
    return {
        collection: row_count * bytes_per_row
        for collection, (row_count, bytes_per_row) in rows.items()
    }


def total_bytes(rebuilds: int | None = None) -> int:
    """All collections plus the index overhead."""
    return int(sum(steady_state_bytes(rebuilds=rebuilds).values()) * (1 + INDEX_OVERHEAD_FRACTION))


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
    # Batch 4 does not change retention: the predictions TTL stays 90 d.
    assert RETENTION_DAYS["weather_predictions"] == 90


def test_live_city_count_covers_the_producer_registry():
    """The budget uses the measured 15 live stations, not the 14 seeded ones."""
    assert PRODUCER_CITY_COUNT == len(eval_python_constant(PRODUCER_SOURCE, "FALLBACK_CITIES"))
    assert LIVE_CITY_COUNT == 15
    assert LIVE_CITY_COUNT > PRODUCER_CITY_COUNT  # Pasadena gap, separate follow-up


def test_horizons_come_from_spark_config():
    """M2 ``target_horizons`` is the source of truth (frozen until WS-A lands)."""
    config = eval_python_constant(SPARK_CONFIG_SOURCE, "FEATURES_CONFIG")
    if "target_horizons" in config:
        assert HORIZONS == _normalize_horizons(config["target_horizons"])
    assert HORIZONS == (1, 3, 6, 12, 24)


def test_steady_state_is_under_cap_with_safety_margin():
    total = total_bytes()
    budget = ATLAS_M0_CAP_BYTES * SAFETY_MARGIN
    assert total < ATLAS_M0_CAP_BYTES, f"{total / MB:.1f} MiB exceeds the {ATLAS_M0_CAP_MB} MiB cap"
    assert (
        total < budget
    ), f"{total / MB:.1f} MiB exceeds the {SAFETY_MARGIN:.0%} budget {budget / MB:.1f} MiB"


def test_prediction_volume_uses_the_structural_rebuild_bound():
    """5 horizons per rebuild, bounded by the rebuild cadence — not hourly."""
    assert prediction_rows_per_day() == FEATURE_REBUILDS_PER_DAY * LIVE_CITY_COUNT * len(HORIZONS)
    assert prediction_rows_per_day() == 8 * 15 * 5 == 600
    # The hourly inference schedule is physically impossible with the upsert.
    assert prediction_rows_per_day() != HOURS_PER_DAY * LIVE_CITY_COUNT * len(HORIZONS)
    expected = (
        RETENTION_DAYS["weather_predictions"]
        * prediction_rows_per_day()
        * MEASURED_AVG_OBJ_SIZE["weather_predictions"]
    )
    assert steady_state_bytes()["weather_predictions"] == expected


def test_budget_tripwire_is_the_feature_rebuild_rate():
    """Sensitivity: the model fails at 14 rebuilds/day, not vacuously.

    Documents the exact boundary — 13/day still fits (≈383.5 MiB), 14/day
    breaches the 75% budget (≈386.4 MiB).
    """
    budget = ATLAS_M0_CAP_BYTES * SAFETY_MARGIN
    assert total_bytes() < budget
    assert total_bytes(rebuilds=13) < budget
    assert (
        total_bytes(rebuilds=14) > budget
    ), f"14 rebuilds/day must breach: {total_bytes(rebuilds=14) / MB:.1f} MiB"


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
