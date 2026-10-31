"""
sync_to_atlas.py
================
Syncs local MongoDB data (raw_weather, weather_features, weather_predictions)
to MongoDB Atlas Free Tier (M0 - 512 MB, no credit card required).

Usage:
  1. Create a free cluster at https://www.mongodb.com/cloud/atlas
  2. Add database user + IP whitelist (0.0.0.0/0 for cloud access).
  3. The Atlas SRV string is MONGO_URI in the single repo-root .env (that is
     also what the deployed app uses); the *source* is the local docker mongo,
     built from MONGO_USERNAME/MONGO_PASSWORD. Override either end for a one-off:
     LOCAL_MONGO_URI=... ATLAS_MONGO_URI=... python scripts/sync_to_atlas.py
  4. Run: python scripts/sync_to_atlas.py (or `make sync-atlas`)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

#: The single env file (2026-10-31: .env / .env.production / atlas-credentials.env
#: were unified). ``MONGO_URI`` in it is the Atlas URI the deployed app uses.
ENV_FILE = Path(".env")

load_dotenv(ENV_FILE)


def local_mongo_url() -> str:
    """The sync source: a local mongo, never Atlas (that would be a self-copy).

    ``LOCAL_MONGO_URI`` wins; otherwise the docker-compose mongo, assembled from
    the same ``MONGO_USERNAME``/``MONGO_PASSWORD`` compose uses. The old default
    came from ``MONGO_URI``, which after the env unification points at Atlas.
    """
    explicit = os.getenv("LOCAL_MONGO_URI")
    if explicit:
        return explicit
    user = os.getenv("MONGO_USERNAME", "admin")
    password = os.getenv("MONGO_PASSWORD", "weatherpass123")
    url = f"mongodb://{user}:{password}@localhost:27017/weather_db?authSource=admin"
    if "@mongodb:27017" in url and not os.path.exists("/.dockerenv"):
        url = url.replace("@mongodb:27017", "@localhost:27017")
    return url


LOCAL_MONGO_URL = local_mongo_url()


def atlas_target() -> str | None:
    """Atlas URI: explicit ATLAS_MONGO_* override, else ``MONGO_URI`` from the environment.

    ``load_dotenv(ENV_FILE)`` has already put the single ``.env``'s ``MONGO_URI``
    in the environment (and a real env var still wins), so no file parsing is
    needed here.
    """
    return (
        os.getenv("ATLAS_MONGO_URI")
        or os.getenv("ATLAS_MONGO_URL")
        or os.getenv("MONGO_URI")
        or os.getenv("MONGO_URL")
    )


ATLAS_MONGO_URL = atlas_target()

COLLECTIONS = ["raw_weather", "weather_features", "weather_predictions"]


def sync_collection_to_atlas(local_db, atlas_db, collection_name: str):
    local_coll = local_db[collection_name]
    atlas_coll = atlas_db[collection_name]

    count = local_coll.count_documents({})
    if count == 0:
        print(f"No documents in local collection '{collection_name}'. Skipping.")
        return

    print(f"Syncing {count} documents from local '{collection_name}' to MongoDB Atlas...")

    bulk_operations = []
    for doc in local_coll.find():
        doc_id = doc.get("_id")
        if doc_id:
            bulk_operations.append(UpdateOne({"_id": doc_id}, {"$set": doc}, upsert=True))

    if bulk_operations:
        result = atlas_coll.bulk_write(bulk_operations)
        print(
            f"Atlas sync for '{collection_name}': "
            f"matched={result.matched_count}, upserted={len(result.upserted_ids)}"
        )


def main():
    if (
        not ATLAS_MONGO_URL
        or "cluster0" not in ATLAS_MONGO_URL
        and "mongodb+srv" not in ATLAS_MONGO_URL
    ):
        print("----------------------------------------------------------------------")
        print("ATTENTION: MongoDB Atlas connection string required!")
        print("1. Go to https://www.mongodb.com/cloud/atlas (Free M0 Cluster, No Credit Card).")
        print("2. Create Database User & allow access IP (0.0.0.0/0).")
        print("3. Set the Atlas SRV string as MONGO_URI in .env (or ATLAS_MONGO_URI):")
        print("   MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/weather_db")
        print("4. Re-run: python scripts/sync_to_atlas.py (or `make sync-atlas`)")
        print("----------------------------------------------------------------------")
        sys.exit(1)

    print("Connecting to Local MongoDB...")
    local_client = MongoClient(LOCAL_MONGO_URL)
    local_db = local_client["weather_db"]

    print("Connecting to MongoDB Atlas...")
    atlas_client = MongoClient(ATLAS_MONGO_URL)
    atlas_db = atlas_client["weather_db"]

    for coll in COLLECTIONS:
        sync_collection_to_atlas(local_db, atlas_db, coll)

    local_client.close()
    atlas_client.close()
    print("MongoDB Atlas sync complete!")


if __name__ == "__main__":
    main()
