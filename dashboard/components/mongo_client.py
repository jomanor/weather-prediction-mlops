import os

import pandas as pd
import streamlit as st
from pymongo import MongoClient


@st.cache_resource
def get_mongo_client():
    mongo_url = os.getenv("MONGO_URI") or os.getenv(
        "MONGO_URL", "mongodb://admin:admin_password@localhost:27017/weather_db?authSource=admin"
    )
    return MongoClient(mongo_url)


def fetch_latest_predictions(city: str = None):
    client = get_mongo_client()
    db = client["weather_db"]
    query = {} if not city else {"city": city}
    cursor = db.weather_predictions.find(query).sort("prediction_timestamp", -1).limit(100)
    data = list(cursor)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


def fetch_observed_weather(city: str, hours: int = 48):
    client = get_mongo_client()
    db = client["weather_db"]
    query = {"city": city}

    # Query raw_weather collection (Open-Meteo payload)
    cursor = db.raw_weather.find(query).sort("timestamp", -1).limit(hours)
    data = list(cursor)

    if not data:
        # Fallback to weather_data if raw_weather is empty
        cursor = db.weather_data.find(query).sort("timestamp", -1).limit(hours)
        data = list(cursor)
        if not data:
            return pd.DataFrame()

    records = []
    for doc in data:
        payload = doc.get("payload", {})
        curr = payload.get("current", {})

        # Flatten payload.current fields
        t = doc.get("timestamp")
        temp = curr.get("temperature_2m", doc.get("temperature"))
        feels = curr.get("apparent_temperature", doc.get("feels_like", temp))
        hum = curr.get("relative_humidity_2m", doc.get("humidity"))
        press = curr.get("surface_pressure", curr.get("pressure_msl", doc.get("pressure")))
        wind = curr.get("wind_speed_10m", doc.get("wind_speed"))
        precip = curr.get("precipitation", doc.get("precipitation", 0.0))
        cloud = curr.get("cloud_cover", doc.get("cloud_cover", 0))

        if t and temp is not None:
            records.append(
                {
                    "city": doc.get("city"),
                    "timestamp": t,
                    "temperature": float(temp),
                    "feels_like": float(feels) if feels is not None else float(temp),
                    "humidity": float(hum) if hum is not None else 0.0,
                    "pressure": float(press) if press is not None else 1013.25,
                    "wind_speed": float(wind) if wind is not None else 0.0,
                    "precipitation": float(precip) if precip is not None else 0.0,
                    "cloud_cover": float(cloud) if cloud is not None else 0,
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp", ascending=True)
    return df
