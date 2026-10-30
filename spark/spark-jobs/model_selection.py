"""Stage-aware selection over the ``model_registry`` collection (Batch 5 / M5).

Every training run lands in ``staging`` (``ml_training.save_model``); only a
human promote (``scripts/promote_model.py``) moves a document to
``production``.  This module is the single place that answers "which registry
document does scoring use", shared by ``inference.py`` and the promote script so
the two can never disagree about what production means.

Rules
-----
- The artifact and metadata source is GridFS + ``model_registry`` (M6).  MLflow
  is never consulted here: this module only *selects* documents.
- Selection is ``find_one`` sorted strictly by ``[("timestamp", -1)]``.  There is
  deliberately **no new index** (the collection holds a handful of docs per
  training run) and adding a global ``{timestamp: -1}`` index is forbidden by
  ``docs/handoff.md``.
- A missing production document is not fatal: ``select_entry`` falls back to the
  newest staging document and reports ``is_fallback=True`` so the caller logs a
  one-line warning.  That keeps inference working between a merge and the first
  promote.

Pure stdlib + pymongo-API-shaped objects: no pyspark, no pymongo import, no
network.  Unit tests drive it with a fake collection.
"""

from __future__ import annotations

import re

REGISTRY_COLLECTION = "model_registry"

STAGE_STAGING = "staging"
STAGE_PRODUCTION = "production"
STAGE_SUPERSEDED = "superseded"

#: Strictly ``timestamp`` descending — the only ordering the registry contract allows.
SORT: list[tuple[str, int]] = [("timestamp", -1)]


def registry_collection(db):
    """The ``model_registry`` collection for *db* (pymongo ``Database`` or a fake)."""
    return db[REGISTRY_COLLECTION]


def prefix_query(
    prefix: str,
    stage: str | None = None,
    exclude_id=None,
) -> dict:
    """Query for docs whose ``model_name`` starts with *prefix*.

    *prefix* may be a family prefix (``temp_prediction_1h_``) or a full model
    name (``temp_prediction_1h_GradientBoostedTrees``); it is regex-escaped so a
    name can never smuggle a pattern.  *stage* adds an exact stage filter and
    *exclude_id* removes one document (used to keep a shadow candidate distinct
    from the production one).
    """
    query: dict = {"model_name": {"$regex": f"^{re.escape(prefix)}"}}
    if stage is not None:
        query["stage"] = stage
    if exclude_id is not None:
        query["_id"] = {"$ne": exclude_id}
    return query


def newest_matching(db, prefix: str, stage: str, exclude_id=None) -> dict | None:
    """Newest document with *stage* whose name starts with *prefix*, or None."""
    return registry_collection(db).find_one(
        prefix_query(prefix, stage, exclude_id),
        sort=list(SORT),
    )


def production_entry(db, prefix: str) -> dict | None:
    """Newest ``stage="production"`` document matching ``^prefix``, or None."""
    return newest_matching(db, prefix, STAGE_PRODUCTION)


def shadow_entry(db, prefix: str, production) -> dict | None:
    """Newest ``stage="staging"`` document **newer than production** for ``^prefix``.

    *production* is the production document (or its ``timestamp``).  A challenger
    must be strictly newer than the incumbent: an older staging artifact was
    already considered (or predates the promoted model) and shadowing it would
    report a stale comparison forever.  The production ``_id`` is excluded as
    belt-and-braces — a shadow is always a *different* model.

    Returns ``None`` when there is no production timestamp to compare against or
    no newer staging document.
    """
    if isinstance(production, dict):
        production_ts = production.get("timestamp")
        production_id = production.get("_id")
    else:
        production_ts = production
        production_id = None
    if production_ts is None:
        return None

    query = prefix_query(prefix, STAGE_STAGING, exclude_id=production_id)
    query["timestamp"] = {"$gt": production_ts}
    return registry_collection(db).find_one(query, sort=list(SORT))


def select_entry(db, prefix: str) -> tuple[dict | None, bool]:
    """Resolve the document inference should score with.

    Returns ``(entry, is_fallback)``:

    - production present -> ``(production_doc, False)``
    - no production but a staging doc -> ``(newest_staging_doc, True)``; the
      caller must log a one-line warning because production is missing
    - nothing at all -> ``(None, False)`` (the caller skips the horizon)

    ``is_fallback`` is only ever True together with a non-None entry.
    """
    entry = production_entry(db, prefix)
    if entry is not None:
        return entry, False

    entry = newest_matching(db, prefix, STAGE_STAGING)
    if entry is not None:
        return entry, True

    return None, False
