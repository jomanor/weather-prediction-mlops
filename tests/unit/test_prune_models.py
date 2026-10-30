"""Retention for the Batch-5 ``model_shadow`` A/B collection.

``scripts/prune_models.py`` already prunes GridFS artifacts + registry docs;
this covers the additive half: shadow A/B documents are bounded by construction
(one per production/shadow pair, never per prediction) and are aged out so a
pair whose models were pruned does not linger.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from tests.conftest import REPO_ROOT
from tests.unit.fake_registry import FakeDb, FakeRegistryCollection

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prune_models as prune  # noqa: E402

NOW = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def _shadow_db(*ages_days: int) -> FakeDb:
    docs = [
        {"_id": f"p{i}_s{i}", "n_samples": 1, "updated_at": NOW - timedelta(days=age)}
        for i, age in enumerate(ages_days)
    ]
    return FakeDb({prune.SHADOW: FakeRegistryCollection(docs)})


def test_prune_shadow_docs_drops_only_stale_pairs():
    db = _shadow_db(1, 29, 31, 90)

    removed = prune.prune_shadow_docs(db, keep_days=30, now=NOW)

    assert removed == 2
    assert sorted(doc["_id"] for doc in db[prune.SHADOW].docs) == ["p0_s0", "p1_s1"]


def test_prune_shadow_docs_is_idempotent():
    db = _shadow_db(31)
    assert prune.prune_shadow_docs(db, keep_days=30, now=NOW) == 1
    assert prune.prune_shadow_docs(db, keep_days=30, now=NOW) == 0


def test_prune_never_touches_the_registry_or_predictions():
    db = _shadow_db(400)
    db["model_registry"] = FakeRegistryCollection([{"_id": "m1", "stage": "production"}])
    db["weather_predictions"] = FakeRegistryCollection([{"_id": "r1"}])

    prune.prune_shadow_docs(db, keep_days=30, now=NOW)

    assert db["model_registry"].docs == [{"_id": "m1", "stage": "production"}]
    assert db["weather_predictions"].docs == [{"_id": "r1"}]
