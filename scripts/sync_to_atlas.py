"""
sync_to_atlas.py
================
Syncs local MongoDB data (raw_weather, weather_features, weather_predictions)
to MongoDB Atlas Free Tier (M0 - 512 MB, no credit card required).

Usage:
  1. Create a free cluster at https://www.mongodb.com/cloud/atlas
  2. Add database user + IP whitelist (0.0.0.0/0 for cloud access).
  3. Set ATLAS_MONGO_URL in .env:
     ATLAS_MONGO_URL=mongodb+srv://<user>:<password>@cluster0.xxx.mongodb.net/weather_db?retryWrites=true&w=majority
  4. Run: python scripts/sync_to_atlas.py
"""

import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

load_dotenv()
if not (os.getenv("ATLAS_MONGO_URI") or os.getenv("ATLAS_MONGO_URL")) and os.path.exists(
    ".env.production"
):
    load_dotenv(".env.production")

LOCAL_MONGO_URL = os.getenv("MONGO_URI") or os.getenv(
    "MONGO_URL", "mongodb://admin:weatherpass123@localhost:27017/weather_db?authSource=admin"
)
if "@mongodb:27017" in LOCAL_MONGO_URL and not os.path.exists("/.dockerenv"):
    LOCAL_MONGO_URL = LOCAL_MONGO_URL.replace("@mongodb:27017", "@localhost:27017")
ATLAS_MONGO_URL = os.getenv("ATLAS_MONGO_URI") or os.getenv("ATLAS_MONGO_URL")

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
        print("3. Add connection string to .env:")
        print("   ATLAS_MONGO_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/weather_db")
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
