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
from tests.unit.fake_registry import FakeDb, FakeRegistryCollection, dt

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


# ---------------------------------------------------------------------------
# Stage-aware artifact retention (QA blocker 1)
# ---------------------------------------------------------------------------

FAMILY = "temp_prediction_1h_GBTRegressor"


def _artifact(file_id: str, day: int) -> dict:
    return {
        "_id": file_id,
        "filename": f"{FAMILY}_202609{day:02d}_000000.zip",
        "uploadDate": dt(day),
    }


def _registry_doc(file_id: str, stage: str, day: int = 1) -> dict:
    return {
        "_id": f"reg-{file_id}",
        "model_name": FAMILY,
        "stage": stage,
        "timestamp": dt(day),
        "gridfs_file_id": file_id,
    }


class _FakeFS:
    """GridFS stand-in: ``delete`` removes the matching ``fs.files`` document."""

    def __init__(self, files: FakeRegistryCollection):
        self.files = files
        self.deleted: list[str] = []

    def delete(self, file_id):
        self.files.docs = [doc for doc in self.files.docs if doc["_id"] != file_id]
        self.deleted.append(file_id)


def _artifact_db(files, registry, shadow=None) -> tuple[FakeDb, _FakeFS]:
    files_coll = FakeRegistryCollection(files)
    db = FakeDb(
        {
            "fs.files": files_coll,
            prune.REGISTRY: FakeRegistryCollection(registry),
            prune.SHADOW: FakeRegistryCollection(shadow or []),
        }
    )
    return db, _FakeFS(files_coll)


def _file_ids(db) -> set[str]:
    return {doc["_id"] for doc in db["fs.files"].docs}


def _registry_ids(db) -> set[str]:
    return {doc["_id"] for doc in db[prune.REGISTRY].docs}


def test_production_artifact_older_than_keep_survives():
    # QA repro: day1 production + day2 + day3, KEEP=2. Production must survive
    # even though it is the oldest artifact in the family.
    db, fs = _artifact_db(
        [_artifact("f1", 1), _artifact("f2", 2), _artifact("f3", 3)],
        [
            _registry_doc("f1", "production", day=1),
            _registry_doc("f2", "staging", day=2),
            _registry_doc("f3", "staging", day=3),
        ],
    )

    removed = prune.prune_collections(db, fs, keep=2, now=NOW)

    assert removed == 0
    assert _file_ids(db) == {"f1", "f2", "f3"}
    assert _registry_ids(db) == {"reg-f1", "reg-f2", "reg-f3"}


def test_superseded_artifact_survives_prune():
    # Superseded is rollback history and is never aged out by KEEP.
    db, fs = _artifact_db(
        [_artifact("f1", 1), _artifact("f2", 2), _artifact("f3", 3), _artifact("f4", 4)],
        [
            _registry_doc("f1", "superseded", day=1),
            _registry_doc("f2", "staging", day=2),
            _registry_doc("f3", "staging", day=3),
            _registry_doc("f4", "staging", day=4),
        ],
    )

    removed = prune.prune_collections(db, fs, keep=2, now=NOW)

    assert removed == 1
    assert _file_ids(db) == {"f1", "f3", "f4"}
    assert "reg-f1" in _registry_ids(db)
    assert "reg-f2" not in _registry_ids(db)


def test_staging_beyond_keep_is_removed_and_count_is_exact():
    db, fs = _artifact_db(
        [_artifact(f"f{day}", day) for day in range(1, 5)],
        [_registry_doc(f"f{day}", "staging", day=day) for day in range(1, 5)],
    )

    removed = prune.prune_collections(db, fs, keep=2, now=NOW)

    assert removed == 2
    assert fs.deleted == ["f2", "f1"]  # oldest first
    assert _file_ids(db) == {"f3", "f4"}
    assert _registry_ids(db) == {"reg-f3", "reg-f4"}

    # Idempotent: a re-run deletes nothing more.
    assert prune.prune_collections(db, fs, keep=2, now=NOW) == 0
    assert _file_ids(db) == {"f3", "f4"}


def test_fresh_orphaned_shadow_doc_is_removed_but_a_live_pair_survives():
    db, fs = _artifact_db(
        [_artifact("f1", 1), _artifact("f2", 2), _artifact("f3", 3)],
        [
            _registry_doc("f1", "staging", day=1),
            _registry_doc("f2", "staging", day=2),
            _registry_doc("f3", "staging", day=3),
        ],
        shadow=[
            # References f1, which this prune deletes. Fresh update must not save it.
            {"_id": "reg-f1_reg-f3", "n_samples": 3, "updated_at": NOW},
            # Both models stay live -> the pair survives.
            {"_id": "reg-f2_reg-f3", "n_samples": 5, "updated_at": NOW},
        ],
    )

    assert prune.prune_collections(db, fs, keep=2, now=NOW) == 1

    shadow_ids = {doc["_id"] for doc in db[prune.SHADOW].docs}
    assert shadow_ids == {"reg-f2_reg-f3"}
