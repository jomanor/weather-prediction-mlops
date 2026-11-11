"""Deduplicate ``weather_predictions`` by (city, source_timestamp, horizon_hours).

``inference.py`` now upserts on a stable ``_id``, but the collection may still
hold duplicates written by the old append-only version. Group the documents by
their business key, keep the one with the newest ``prediction_timestamp``, and
delete the rest.

Dry-run by default; pass ``--apply`` to actually delete. Safe to re-run: after
a successful pass every group has a single document, so subsequent runs report
zero duplicates.

    MONGO_URI="mongodb+srv://..." python scripts/dedupe_predictions.py
    MONGO_URI="mongodb+srv://..." python scripts/dedupe_predictions.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from pymongo import MongoClient

DB_NAME = "weather_db"
COLLECTION = "weather_predictions"

#: Only the fields the grouping needs. The full documents (payload, intervals,
#: model metadata) never leave Mongo: on a 512 MB M0 an unpaged ``find({})``
#: materialising every document is what exhausted the instance (audit finding 20).
PROJECTION = {
    "_id": 1,
    "city": 1,
    "source_timestamp": 1,
    "horizon_hours": 1,
    "prediction_timestamp": 1,
}

#: Duplicates are deleted in ``$in`` batches of this size instead of one command
#: carrying the whole collection.
DELETE_BATCH_SIZE = 1000

#: Sort key for a document missing ``prediction_timestamp``: tz-aware so it can
#: compare against the tz-aware datetimes PyMongo returns, and the earliest
#: possible instant so such a document is always the one deleted.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _newest_key(doc: dict) -> datetime:
    ts = doc.get("prediction_timestamp")
    return ts if isinstance(ts, datetime) else _EPOCH


def _duplicate_ids(db) -> list:
    """Return the _id of every duplicate beyond the newest document per group.

    One streaming pass over a projection of the key fields: memory is bounded by
    the number of *groups* (newest _id each) plus one _id per duplicate, never by
    the size of the documents.
    """
    newest: dict[tuple, tuple[datetime, object]] = {}
    to_delete: list = []
    for doc in db[COLLECTION].find({}, PROJECTION):
        identity = doc.get("_id")
        if identity is None:
            continue
        key = (doc.get("city"), doc.get("source_timestamp"), doc.get("horizon_hours"))
        timestamp = _newest_key(doc)
        current = newest.get(key)
        if current is None:
            newest[key] = (timestamp, identity)
        elif timestamp > current[0]:
            newest[key] = (timestamp, identity)
            to_delete.append(current[1])
        else:
            # Older, or the same instant: the first document seen wins, exactly
            # like the previous ``sort(key=..., reverse=True)`` behaviour.
            to_delete.append(identity)
    return to_delete


def _delete_in_batches(collection, ids: list) -> int:
    """Delete ``ids`` in ``$in`` batches; returns the number the collection removed."""
    deleted = 0
    for start in range(0, len(ids), DELETE_BATCH_SIZE):
        batch = ids[start : start + DELETE_BATCH_SIZE]
        deleted += collection.delete_many({"_id": {"$in": batch}}).deleted_count
    return deleted


def dedupe(mongo_uri: str, apply: bool = False) -> tuple[int, int]:
    """Delete duplicate predictions. Returns (duplicate_count, deleted_count)."""
    client = MongoClient(mongo_uri)
    try:
        db = client[DB_NAME]
        to_delete = _duplicate_ids(db)
        duplicate_count = len(to_delete)

        if apply and to_delete:
            return duplicate_count, _delete_in_batches(db[COLLECTION], to_delete)
        return duplicate_count, 0
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete the duplicate documents (default: dry-run only)",
    )
    args = parser.parse_args()

    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
    if not mongo_uri:
        print("MONGO_URI environment variable is not set.", file=sys.stderr)
        return 1

    duplicate_count, deleted = dedupe(mongo_uri, apply=args.apply)

    if duplicate_count:
        if args.apply:
            print(f"deleted {deleted} duplicate prediction document(s)")
            print("weather_predictions is now deduplicated.")
        else:
            print(
                f"{duplicate_count} duplicate prediction document(s) found; "
                "re-run with --apply to delete them"
            )
    else:
        print("no duplicate prediction documents found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
