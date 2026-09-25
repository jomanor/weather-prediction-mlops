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
from collections import defaultdict
from datetime import datetime, timezone

from pymongo import MongoClient

DB_NAME = "weather_db"
COLLECTION = "weather_predictions"

#: Sort key for a document missing ``prediction_timestamp``: tz-aware so it can
#: compare against the tz-aware datetimes PyMongo returns, and the earliest
#: possible instant so such a document is always the one deleted.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _newest_key(doc: dict) -> datetime:
    ts = doc.get("prediction_timestamp")
    return ts if isinstance(ts, datetime) else _EPOCH


def _duplicate_ids(db) -> list:
    """Return the _id of every duplicate beyond the newest document per group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for doc in db[COLLECTION].find({}):
        key = (doc.get("city"), doc.get("source_timestamp"), doc.get("horizon_hours"))
        groups[key].append(doc)

    to_delete: list = []
    for docs in groups.values():
        if len(docs) <= 1:
            continue
        docs.sort(key=_newest_key, reverse=True)
        to_delete.extend(doc["_id"] for doc in docs[1:] if "_id" in doc)
    return to_delete


def dedupe(mongo_uri: str, apply: bool = False) -> tuple[int, int]:
    """Delete duplicate predictions. Returns (duplicate_count, deleted_count)."""
    client = MongoClient(mongo_uri)
    try:
        db = client[DB_NAME]
        to_delete = _duplicate_ids(db)
        duplicate_count = len(to_delete)

        if apply and to_delete:
            result = db[COLLECTION].delete_many({"_id": {"$in": to_delete}})
            return duplicate_count, result.deleted_count
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
