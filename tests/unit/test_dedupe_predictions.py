"""``scripts/dedupe_predictions.py`` scans a projection, not whole documents.

The dedupe pass runs against Atlas M0 (512 MB), so it must never materialise the
collection: it reads only the grouping fields through a cursor and deletes in
batches. Both properties are asserted here, plus the "keep the newest document,
first seen wins on a tie" rule the previous implementation got from a stable sort.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import dedupe_predictions as dedupe  # noqa: E402

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
SOURCE = NOW - timedelta(hours=1)


class _Cursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    def __iter__(self):
        return iter(self._docs)


class _Collection:
    """Stands in for a ``pymongo`` collection: records projections and deletes."""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = list(docs)
        self.projections: list[dict | None] = []
        self.deleted_batches: list[list] = []

    def find(self, query: dict, projection: dict | None = None) -> _Cursor:
        self.projections.append(projection)
        return _Cursor(self.docs)

    def delete_many(self, query: dict):
        ids = set(query["_id"]["$in"])
        self.deleted_batches.append(sorted(ids))
        kept = [doc for doc in self.docs if doc["_id"] not in ids]
        deleted = len(self.docs) - len(kept)
        self.docs = kept
        return SimpleNamespace(deleted_count=deleted)


class _Db(dict):
    def __init__(self, collection: _Collection) -> None:
        super().__init__({dedupe.COLLECTION: collection})


def _row(identity: str, *, created_hours_ago: float, **overrides) -> dict:
    doc = {
        "_id": identity,
        "city": "Madrid",
        "source_timestamp": SOURCE,
        "horizon_hours": 1,
        "prediction_timestamp": NOW - timedelta(hours=created_hours_ago),
    }
    doc.update(overrides)
    return doc


def test_keeps_the_newest_of_each_group_and_deletes_the_rest():
    collection = _Collection(
        [
            _row("old", created_hours_ago=2),
            _row("new", created_hours_ago=0),
            _row("other-city", created_hours_ago=0, city="Alicante"),
        ]
    )

    assert dedupe._duplicate_ids(_Db(collection)) == ["old"]


def test_a_missing_prediction_timestamp_loses_to_a_real_one():
    collection = _Collection(
        [
            _row("undated", created_hours_ago=0, prediction_timestamp=None),
            _row("dated", created_hours_ago=48),
        ]
    )

    assert dedupe._duplicate_ids(_Db(collection)) == ["undated"]


def test_equal_timestamps_keep_the_first_document_seen():
    collection = _Collection(
        [_row("first", created_hours_ago=0), _row("second", created_hours_ago=0)]
    )

    assert dedupe._duplicate_ids(_Db(collection)) == ["second"]


def test_the_scan_projects_only_the_grouping_fields():
    collection = _Collection([_row("only", created_hours_ago=0)])

    dedupe._duplicate_ids(_Db(collection))

    assert set(collection.projections[0]) == {
        "_id",
        "city",
        "source_timestamp",
        "horizon_hours",
        "prediction_timestamp",
    }


def test_duplicates_are_deleted_in_batches(monkeypatch):
    monkeypatch.setattr(dedupe, "DELETE_BATCH_SIZE", 2)
    collection = _Collection(
        [_row("keep", created_hours_ago=0)]
        + [_row(f"dup{index}", created_hours_ago=1) for index in range(5)]
    )
    ids = dedupe._duplicate_ids(_Db(collection))

    assert dedupe._delete_in_batches(collection, ids) == 5
    assert [len(batch) for batch in collection.deleted_batches] == [2, 2, 1]
    assert [doc["_id"] for doc in collection.docs] == ["keep"]
