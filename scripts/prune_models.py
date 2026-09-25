"""Prune old Spark model artifacts from Atlas.

`ml_training.save_model` pushes a zipped PipelineModel to GridFS on every run
and inserts a matching `model_registry` document. Nothing ever removes them, so
a daily retrain adds roughly 1-2 MB per day against the 512 MB MongoDB Atlas
free tier. `inference.py` only ever loads the newest model for each prefix, so
everything older is dead weight.

This keeps the newest `KEEP` artifacts per model family and deletes the rest.
Run it after training, or standalone:

    MONGO_URI="mongodb+srv://..." python scripts/prune_models.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import gridfs
from pymongo import MongoClient

# GridFS filenames look like "temp_prediction_1h_GBTRegressor_20260924_024501.zip".
# Group by everything except the trailing timestamp.
MODEL_FILENAME = re.compile(r"^(?P<family>(?:temp|rain)_prediction_\d+h_.+)_\d{8}_\d{6}\.zip$")
FILENAME_FILTER = {"$regex": "^(temp|rain)_prediction_"}

KEEP = int(os.getenv("KEEP_MODELS", "2"))
DB_NAME = "weather_db"
REGISTRY = "model_registry"


def prune(mongo_uri: str, keep: int = KEEP) -> int:
    """Delete all but the newest `keep` artifacts per family. Returns the count."""
    client = MongoClient(mongo_uri)
    try:
        db = client[DB_NAME]
        fs = gridfs.GridFS(db)

        families: dict[str, list[dict]] = defaultdict(list)
        for doc in db["fs.files"].find({"filename": FILENAME_FILTER}):
            match = MODEL_FILENAME.match(doc["filename"])
            if match:
                families[match.group("family")].append(doc)

        removed = 0
        for family, docs in sorted(families.items()):
            docs.sort(key=lambda d: d["uploadDate"], reverse=True)
            for doc in docs[keep:]:
                try:
                    fs.delete(doc["_id"])
                except Exception as exc:  # noqa: BLE001 - a stuck file must not fail the run
                    print(f"warn: could not delete {doc['filename']}: {exc}")
                    continue
                db[REGISTRY].delete_many({"gridfs_file_id": doc["_id"]})
                removed += 1
            print(f"{family}: kept {min(len(docs), keep)}/{len(docs)}")

        return removed
    finally:
        client.close()


def main() -> int:
    mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
    if not mongo_uri:
        print("MONGO_URI environment variable is not set.", file=sys.stderr)
        return 1

    removed = prune(mongo_uri)
    print(f"pruned {removed} model artifact(s), keeping the newest {KEEP} per family")
    # GitHub Actions workflow command: surfaces the count as a run notice while
    # remaining ordinary stdout everywhere else (local runs, docker compose).
    print(f"::notice::prune ran: {removed} deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
