"""In-memory ``model_registry`` fake shared by the Batch-5 unit tests.

No pymongo, no network: only the handful of pymongo collection operations the
selection module, the promote script and the shadow A/B path actually use —
``find_one`` (with ``sort``), ``find`` (with ``sort``), ``update_one`` (``$set``
/ ``$unset``) and ``replace_one(..., upsert=True)``.

Query support is deliberately small and explicit so a test cannot pass on a
query the real code could not run: equality, ``{"$ne": ...}`` on ``_id`` and
``{"$regex": "^literal"}`` on ``model_name``.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone


def _match_value(actual, expected) -> bool:
    if isinstance(expected, dict):
        if "$ne" in expected:
            return actual != expected["$ne"]
        if "$lt" in expected:
            return actual is not None and actual < expected["$lt"]
        if "$gt" in expected:
            return actual is not None and actual > expected["$gt"]
        if "$gte" in expected:
            return actual is not None and actual >= expected["$gte"]
        if "$regex" in expected:
            return isinstance(actual, str) and re.search(expected["$regex"], actual) is not None
        raise AssertionError(f"FakeRegistryCollection: unsupported query operator {expected}")
    if isinstance(actual, datetime) and isinstance(expected, datetime):
        return actual == expected
    return actual == expected


def _matches(doc: dict, query: dict) -> bool:
    for key, expected in query.items():
        if isinstance(expected, dict) and "$exists" in expected:
            if (key in doc) != bool(expected["$exists"]):
                return False
            continue
        if key not in doc and not (isinstance(expected, dict) and "$ne" in expected):
            return False
        if not _match_value(doc.get(key), expected):
            return False
    return True


class FakeRegistryCollection:
    """Minimal pymongo-collection stand-in backed by a list of dicts."""

    def __init__(self, docs: list[dict] | None = None):
        self.docs = [copy.deepcopy(doc) for doc in (docs or [])]
        self.unsets: list[tuple[dict, dict]] = []
        #: ``(query, update)`` in order — lets a test assert write ordering.
        self.updates: list[tuple[dict, dict]] = []

    # -- reads ---------------------------------------------------------
    def _sorted(self, matches: list[dict], sort) -> list[dict]:
        if not sort:
            return matches
        ordered = list(matches)
        # Apply keys from last to first so multi-key sorts behave like Mongo.
        for key, direction in reversed(list(sort)):
            ordered.sort(key=lambda doc: (doc.get(key) is None, doc.get(key)), reverse=direction < 0)
        return ordered

    def find_one(self, query: dict, sort=None, **kwargs):
        matches = self._sorted([doc for doc in self.docs if _matches(doc, query)], sort)
        return copy.deepcopy(matches[0]) if matches else None

    def find(self, query: dict | None = None, sort=None, **kwargs):
        matches = self._sorted(
            [doc for doc in self.docs if _matches(doc, query or {})], sort
        )
        return [copy.deepcopy(doc) for doc in matches]

    # -- writes --------------------------------------------------------
    def _find_ref(self, query: dict) -> dict | None:
        for doc in self.docs:
            if _matches(doc, query):
                return doc
        return None

    def update_one(self, query: dict, update: dict, **kwargs):
        self.updates.append((copy.deepcopy(query), copy.deepcopy(update)))
        doc = self._find_ref(query)
        if doc is None:
            return _UpdateResult(0, 0)
        for key, value in (update.get("$set") or {}).items():
            doc[key] = value
        if update.get("$unset"):
            self.unsets.append((query, dict(update["$unset"])))
            for key in update["$unset"]:
                doc.pop(key, None)
        return _UpdateResult(1, 1)

    def delete_many(self, query: dict):
        before = len(self.docs)
        self.docs = [doc for doc in self.docs if not _matches(doc, query)]
        removed = before - len(self.docs)
        return _UpdateResult(removed, removed, removed)

    def replace_one(self, query: dict, replacement: dict, upsert: bool = False):
        doc = self._find_ref(query)
        if doc is None:
            if not upsert:
                return _UpdateResult(0, 0)
            self.docs.append(copy.deepcopy(replacement))
            return _UpdateResult(0, 1)
        doc.clear()
        doc.update(copy.deepcopy(replacement))
        return _UpdateResult(1, 1)


class _UpdateResult:
    """pymongo-shaped result (``delete_many`` reports ``deleted_count`` only)."""

    def __init__(self, matched_count: int, modified_count: int, deleted_count: int = 0):
        self.matched_count = matched_count
        self.modified_count = modified_count
        self.deleted_count = deleted_count


class FakeDb:
    """``db[collection]`` -> :class:`FakeRegistryCollection`."""

    def __init__(self, collections: dict[str, FakeRegistryCollection] | None = None):
        self._collections = dict(collections or {})

    def __getitem__(self, name: str) -> FakeRegistryCollection:
        return self._collections.setdefault(name, FakeRegistryCollection())

    def __setitem__(self, name: str, collection: FakeRegistryCollection) -> None:
        self._collections[name] = collection

    @property
    def collections(self) -> dict[str, FakeRegistryCollection]:
        return self._collections


def dt(day: int, hour: int = 0) -> datetime:
    """Convenience UTC timestamp for registry docs (September 2026)."""
    return datetime(2026, 9, day, hour, 0, 0, tzinfo=timezone.utc)


def make_entry(
    model_name: str,
    *,
    doc_id: str,
    stage: str = "staging",
    horizon: int = 1,
    target: str = "temperature",
    version: str | None = None,
    timestamp: datetime | None = None,
    gridfs_file_id: str | None = "gridfs-id",
    **extra,
) -> dict:
    """One registry document, shaped like ``ml_training.save_model`` output."""
    doc = {
        "_id": doc_id,
        "model_name": model_name,
        "model_type": "GradientBoostedTrees",
        "stage": stage,
        "horizon_hours": horizon,
        "target": target,
        "version": version or f"202609{doc_id[-2:]}_000000",
        "timestamp": timestamp or dt(1),
        "gridfs_file_id": gridfs_file_id,
    }
    doc.update(extra)
    return doc
