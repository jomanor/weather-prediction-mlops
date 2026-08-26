"""
sync_to_supabase.py
===================
Syncs raw weather observations, engineered features, and model predictions
from MongoDB to Supabase (PostgreSQL free cloud tier).

Usage:
  1. Set SUPABASE_URL and SUPABASE_KEY in your .env file.
  2. Run: python scripts/sync_to_supabase.py
"""

import os
import sys

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MONGO_URL = os.getenv("MONGO_URI") or os.getenv(
    "MONGO_URL", "mongodb://admin:weatherpass123@localhost:27017/weather_db?authSource=admin"
)
if "@mongodb:27017" in MONGO_URL and not os.path.exists("/.dockerenv"):
    MONGO_URL = MONGO_URL.replace("@mongodb:27017", "@localhost:27017")


def sync_collection(db, collection_name: str):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print(f"ERROR: SUPABASE_URL or SUPABASE_KEY not set in .env. Skipping {collection_name}.")
        return

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    endpoint = f"{SUPABASE_URL}/rest/v1/{collection_name}"
    docs = list(db[collection_name].find().limit(500))

    if not docs:
        print(f"No documents found in MongoDB collection '{collection_name}'.")
        return

    cleaned_records = []
    for doc in docs:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        if "timestamp" in doc and hasattr(doc["timestamp"], "isoformat"):
            doc["timestamp"] = doc["timestamp"].isoformat()
        if "prediction_timestamp" in doc and hasattr(doc["prediction_timestamp"], "isoformat"):
            doc["prediction_timestamp"] = doc["prediction_timestamp"].isoformat()
        if "source_timestamp" in doc and hasattr(doc["source_timestamp"], "isoformat"):
            doc["source_timestamp"] = doc["source_timestamp"].isoformat()
        cleaned_records.append(doc)

    try:
        response = requests.post(endpoint, headers=headers, json=cleaned_records, timeout=10)
        if response.status_code in (200, 201):
            print(
                f"Successfully synced {len(cleaned_records)} records "
                f"to Supabase table '{collection_name}'."
            )
        else:
            print(
                f"Failed to sync {collection_name} to Supabase: "
                f"{response.status_code} - {response.text}"
            )
    except Exception as e:
        print(f"Error syncing {collection_name} to Supabase: {e}")


def main():
    if not SUPABASE_URL or "your-supabase" in SUPABASE_URL:
        print("----------------------------------------------------------------------")
        print("ATTENTION: Supabase project credentials required!")
        print("To push data to your remote Supabase instance:")
        print("1. Go to https://supabase.com and create a free project.")
        print("2. Copy Project URL and Anon Key into .env:")
        print("     SUPABASE_URL=https://<project-ref>.supabase.co")
        print("     SUPABASE_KEY=<your-anon-key>")
        print("3. Re-run: python scripts/sync_to_supabase.py")
        print("----------------------------------------------------------------------")
        sys.exit(1)

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URL)
    db = client["weather_db"]

    sync_collection(db, "raw_weather")
    sync_collection(db, "weather_features")
    sync_collection(db, "weather_predictions")
    client.close()


if __name__ == "__main__":
    main()
