"""Stage-aware registry selection (``spark/spark-jobs/model_selection.py``).

The whole M5 lifecycle hangs off these three functions: inference scores
*production*, the promote script moves staging -> production, and the shadow A/B
needs a challenger distinct from the production document. No Mongo, no Spark —
the fake collection only implements the queries the module issues.
"""

from __future__ import annotations

import os
import sys

from tests.unit.fake_registry import FakeDb, FakeRegistryCollection, dt, make_entry

SPARK_JOBS_DIR = os.path.join(os.path.dirname(__file__), "../../spark/spark-jobs")
if SPARK_JOBS_DIR not in sys.path:
    sys.path.insert(0, SPARK_JOBS_DIR)

import model_selection as ms  # noqa: E402


def _db(*docs) -> FakeDb:
    return FakeDb({ms.REGISTRY_COLLECTION: FakeRegistryCollection(list(docs))})


# ---------------------------------------------------------------------------
# Sorting / query shape
# ---------------------------------------------------------------------------


def test_sort_is_strictly_timestamp_desc():
    # There is deliberately no new index: the registry is tiny and every read
    # must sort on timestamp alone.
    assert ms.SORT == [("timestamp", -1)]


def test_prefix_query_is_escaped():
    query = ms.prefix_query("temp_prediction_1h_", ms.STAGE_PRODUCTION, exclude_id="abc")
    assert query["stage"] == "production"
    assert query["_id"] == {"$ne": "abc"}
    # A dot must be matched literally, not as a regex wildcard.
    assert ms.prefix_query("temp.prediction")["model_name"] == {"$regex": "^temp\\.prediction"}


def test_prefix_does_not_match_a_different_horizon():
    db = _db(
        make_entry("temp_prediction_12h_GBT", doc_id="12", stage="production", horizon=12),
        make_entry("temp_prediction_1h_GBT", doc_id="01", stage="production", horizon=1),
    )
    entry = ms.production_entry(db, "temp_prediction_1h_")
    assert entry is not None
    assert entry["_id"] == "01"


# ---------------------------------------------------------------------------
# production_entry
# ---------------------------------------------------------------------------


def test_production_entry_picks_newest_production():
    db = _db(
        make_entry("temp_prediction_1h_Old", doc_id="01", stage="production", timestamp=dt(1)),
        make_entry("temp_prediction_1h_New", doc_id="02", stage="production", timestamp=dt(3)),
        make_entry("temp_prediction_1h_Staged", doc_id="03", stage="staging", timestamp=dt(4)),
        make_entry("rain_prediction_1h_GBT", doc_id="04", stage="production", timestamp=dt(5)),
    )
    entry = ms.production_entry(db, "temp_prediction_1h_")
    assert entry["_id"] == "02"  # newest production for this prefix only


def test_production_entry_none_when_only_staging():
    db = _db(make_entry("temp_prediction_1h_GBT", doc_id="01", stage="staging"))
    assert ms.production_entry(db, "temp_prediction_1h_") is None


def test_production_entry_ignores_superseded():
    db = _db(
        make_entry("temp_prediction_1h_Old", doc_id="01", stage="superseded", timestamp=dt(5))
    )
    assert ms.production_entry(db, "temp_prediction_1h_") is None


# ---------------------------------------------------------------------------
# shadow_entry
# ---------------------------------------------------------------------------


def test_shadow_entry_picks_newest_staging_excluding_production():
    production = make_entry(
        "temp_prediction_1h_Prod", doc_id="01", stage="production", timestamp=dt(1)
    )
    db = _db(
        production,
        make_entry("temp_prediction_1h_Old", doc_id="02", stage="staging", timestamp=dt(2)),
        make_entry("temp_prediction_1h_New", doc_id="03", stage="staging", timestamp=dt(3)),
        make_entry("temp_prediction_1h_Gone", doc_id="04", stage="superseded", timestamp=dt(4)),
    )
    shadow = ms.shadow_entry(db, "temp_prediction_1h_", production)
    assert shadow["_id"] == "03"
    assert shadow["stage"] == "staging"


def test_shadow_entry_requires_a_challenger_newer_than_production():
    # Docs say "newer staging": an older staging artifact is not a challenger.
    production = make_entry(
        "temp_prediction_1h_Prod", doc_id="01", stage="production", timestamp=dt(9)
    )
    db = _db(
        production,
        make_entry("temp_prediction_1h_Old", doc_id="02", stage="staging", timestamp=dt(2)),
    )
    assert ms.shadow_entry(db, "temp_prediction_1h_", production) is None


def test_shadow_entry_returns_a_newer_staging_challenger():
    production = make_entry(
        "temp_prediction_1h_Prod", doc_id="01", stage="production", timestamp=dt(9)
    )
    db = _db(
        production,
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(10)),
    )
    shadow = ms.shadow_entry(db, "temp_prediction_1h_", production)
    assert shadow is not None and shadow["_id"] == "02"


def test_shadow_entry_none_without_challenger():
    production = make_entry("temp_prediction_1h_Prod", doc_id="01", stage="production")
    db = _db(
        production,
        make_entry("temp_prediction_1h_Gone", doc_id="02", stage="superseded"),
    )
    assert ms.shadow_entry(db, "temp_prediction_1h_", production) is None


def test_shadow_entry_never_returns_the_production_doc():
    # Belt-and-braces: even a doc that matches the staging filter but shares the
    # production _id must be excluded (a shadow is always a different model).
    production = make_entry("temp_prediction_1h_Prod", doc_id="01", stage="staging")
    db = _db(production)
    assert ms.shadow_entry(db, "temp_prediction_1h_", production) is None


def test_shadow_entry_accepts_a_bare_timestamp():
    db = _db(
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(10)),
    )
    shadow = ms.shadow_entry(db, "temp_prediction_1h_", dt(9))
    assert shadow is not None and shadow["_id"] == "02"
    assert ms.shadow_entry(db, "temp_prediction_1h_", dt(10)) is None


# ---------------------------------------------------------------------------
# select_entry
# ---------------------------------------------------------------------------


def test_select_entry_prefers_production_even_when_staging_is_newer():
    db = _db(
        make_entry("temp_prediction_1h_Prod", doc_id="01", stage="production", timestamp=dt(1)),
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(9)),
    )
    entry, is_fallback = ms.select_entry(db, "temp_prediction_1h_")
    assert entry["_id"] == "01"
    assert is_fallback is False


def test_select_entry_falls_back_to_newest_staging_with_flag():
    db = _db(
        make_entry("temp_prediction_1h_Old", doc_id="01", stage="staging", timestamp=dt(2)),
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(5)),
    )
    entry, is_fallback = ms.select_entry(db, "temp_prediction_1h_")
    assert entry["_id"] == "02"
    assert is_fallback is True


def test_select_entry_returns_none_when_nothing_matches():
    db = _db(make_entry("rain_prediction_1h_GBT", doc_id="01", stage="production"))
    entry, is_fallback = ms.select_entry(db, "temp_prediction_1h_")
    assert entry is None
    assert is_fallback is False


def test_select_entry_missing_prefix_is_safe():
    db = _db()
    assert ms.select_entry(db, "temp_prediction_1h_") == (None, False)
