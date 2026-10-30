"""Prune old Spark model artifacts from Atlas.

`ml_training.save_model` pushes a zipped PipelineModel to GridFS on every run
and inserts a matching `model_registry` document. Nothing ever removes them, so
a daily retrain adds roughly 1-2 MB per day against the 512 MB MongoDB Atlas
free tier. `inference.py` only ever loads the newest model for each prefix, so
everything older is dead weight.

Retention is **stage-aware** (Batch 5 lifecycle fix): within each family every
artifact referenced by a registry document whose stage is `production` or
`superseded` is kept unconditionally — production is what inference scores and
`superseded` is the rollback history. The `KEEP` bound only applies to the
remaining (staging/unstaged) artifacts, newest-first. Deleting the production
artifact (QA repro: day1 prod + day2 + day3 with `KEEP=2`) used to silently drop
inference to the staging fallback. GridFS files and their registry documents are
deleted together, and only for artifacts actually deleted.

It also ages out the Batch-5 `model_shadow` A/B documents. That collection is
bounded by construction — one document per (production model, shadow model) pair,
so O(#model-pairs) and never O(#predictions) — but a stale pair from a pruned
model family has no consumer. Two rules apply: a shadow doc whose pair references
a model id that no longer exists in `model_registry` is deleted immediately (a
fresh orphan must not survive just because `updated_at` is recent), and anything
untouched for `KEEP_SHADOW_DAYS` goes.

Run it after training, or standalone:

    MONGO_URI="mongodb+srv://..." python scripts/prune_models.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import gridfs
from pymongo import MongoClient

# GridFS filenames look like "temp_prediction_1h_GBTRegressor_20260924_024501.zip".
# Group by everything except the trailing timestamp.
MODEL_FILENAME = re.compile(r"^(?P<family>(?:temp|rain)_prediction_\d+h_.+)_\d{8}_\d{6}\.zip$")
FILENAME_FILTER = {"$regex": "^(temp|rain)_prediction_"}

KEEP = int(os.getenv("KEEP_MODELS", "2"))
DB_NAME = "weather_db"
REGISTRY = "model_registry"
SHADOW = "model_shadow"
#: Registry stages that always survive a prune: production is served, superseded
#: is rollback history.
PROTECTED_STAGES = frozenset({"production", "superseded"})
#: Shadow A/B docs are refreshed on every inference run while both models live;
#: a pair untouched for this long belongs to a pruned family.
KEEP_SHADOW_DAYS = int(os.getenv("KEEP_SHADOW_DAYS", "30"))


def prune_shadow_docs(db, keep_days: int = KEEP_SHADOW_DAYS, now: datetime | None = None) -> int:
    """Delete `model_shadow` docs not refreshed in *keep_days*. Returns the count."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=keep_days)
    result = db[SHADOW].delete_many({"updated_at": {"$lt": cutoff}})
    return getattr(result, "deleted_count", 0)


def _stage_by_file(db) -> dict[str, str]:
    """``gridfs_file_id`` (as text) -> registry ``stage`` for every registry doc."""
    stages: dict[str, str] = {}
    for doc in db[REGISTRY].find({}):
        file_id = doc.get("gridfs_file_id")
        if file_id is None:
            continue
        stage = doc.get("stage")
        if stage is not None:
            stages[str(file_id)] = stage
    return stages


def prune_orphan_shadow_docs(db) -> int:
    """Delete shadow docs whose ``(production, shadow)`` pair lost a model.

    ``model_shadow._id`` is ``"{production_id}_{shadow_id}"`` (see
    ``inference.shadow_doc_id``).  A pair that references a registry id that no
    longer exists is unusable, no matter how recently it was refreshed.
    """
    live_ids = {str(doc["_id"]) for doc in db[REGISTRY].find({})}
    removed = 0
    for doc in list(db[SHADOW].find({})):
        parts = str(doc.get("_id", "")).split("_")
        if len(parts) < 2:
            continue
        if all(part in live_ids for part in parts):
            continue
        result = db[SHADOW].delete_many({"_id": doc["_id"]})
        removed += getattr(result, "deleted_count", 0)
    return removed


def _split_family(db, docs: list[dict], keep: int) -> tuple[list[dict], list[dict]]:
    """``(survivors, doomed)`` for one family.

    Production/superseded artifacts always survive; the ``keep`` newest of the
    remaining (staging/unstaged) artifacts survive too.
    """
    stages = _stage_by_file(db)
    ordered = sorted(docs, key=lambda d: d["uploadDate"], reverse=True)
    protected = [d for d in ordered if stages.get(str(d["_id"])) in PROTECTED_STAGES]
    protected_ids = {str(d["_id"]) for d in protected}
    # The KEEP bound only bites on staging/unstaged artifacts.
    others = [d for d in ordered if str(d["_id"]) not in protected_ids]
    return protected + others[:keep], others[keep:]


def prune_collections(db, fs, keep: int = KEEP, now: datetime | None = None) -> int:
    """Stage-aware prune over *db*, deleting GridFS files with *fs*.

    Separated from :func:`prune` so unit tests can drive it with the in-memory
    registry fake and a fake GridFS (no Mongo, no network).
    """
    families: dict[str, list[dict]] = defaultdict(list)
    for doc in db["fs.files"].find({"filename": FILENAME_FILTER}):
        match = MODEL_FILENAME.match(doc["filename"])
        if match:
            families[match.group("family")].append(doc)

    removed = 0
    for family, docs in sorted(families.items()):
        survivors, doomed = _split_family(db, docs, keep)
        for doc in doomed:
            try:
                fs.delete(doc["_id"])
            except Exception as exc:  # noqa: BLE001 - a stuck file must not fail the run
                print(f"warn: could not delete {doc['filename']}: {exc}")
                continue
            # Only the registry docs of artifacts actually deleted.
            db[REGISTRY].delete_many({"gridfs_file_id": doc["_id"]})
            removed += 1
        print(f"{family}: kept {len(survivors)}/{len(docs)}")

    orphan_shadow = prune_orphan_shadow_docs(db)
    stale_shadow = prune_shadow_docs(db, now=now)
    print(f"{SHADOW}: removed {orphan_shadow} orphaned and {stale_shadow} stale A/B doc(s)")

    return removed


def prune(mongo_uri: str, keep: int = KEEP) -> int:
    """Delete all but the newest `keep` staging artifacts per family (plus
    production/superseded, which always survive). Returns the count."""
    client = MongoClient(mongo_uri)
    try:
        db = client[DB_NAME]
        fs = gridfs.GridFS(db)
        return prune_collections(db, fs, keep)
    finally:
        client.close()


def main() -> int:
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
    if not mongo_uri:
        print("MONGO_URI environment variable is not set.", file=sys.stderr)
        return 1

    removed = prune(mongo_uri)
    print(
        f"pruned {removed} model artifact(s), keeping production/superseded plus "
        f"the newest {KEEP} staging per family"
    )
    # GitHub Actions workflow command: surfaces the count as a run notice while
    # remaining ordinary stdout everywhere else (local runs, docker compose).
    print(f"::notice::prune ran: {removed} deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
