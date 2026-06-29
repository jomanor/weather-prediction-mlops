import json
import os
from datetime import datetime, timezone
from kafka import KafkaConsumer
from pymongo import MongoClient


class WeatherConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            "weather-data",
            bootstrap_servers=os.getenv("KAFKA_BROKER", "kafka:9092"),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            group_id="weather-consumer-group",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        mongo_url = os.getenv("MONGO_URL")
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client["weather_db"]

        # raw_weather  → everything that arrives from open-meteo, untouched
        # weather_data → only the current snapshot, used by the API
        self.raw_collection = self.db["raw_weather"]
        self.current_collection = self.db["weather_data"]

    def _unix_to_datetime(self, unix_ts):
        """Convert a unix integer timestamp to a UTC-aware datetime."""
        try:
            return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
        except Exception:
            return datetime.now(tz=timezone.utc)

    def process_message(self, message):
        """Store raw Open-Meteo payload and a clean current-conditions snapshot."""
        data = message.value

        city = data.get("city", "unknown")
        current = data.get("current", {})

        # --- timestamp (open-meteo returns unix int when timeformat=unixtime) ---
        timestamp_dt = self._unix_to_datetime(current.get("time", 0))
        timestamp_unix = int(timestamp_dt.timestamp())

        # -----------------------------------------------------------------------
        # 1. Save full raw payload to raw_weather
        # -----------------------------------------------------------------------
        raw_doc = {
            "_id": f"{city}_{timestamp_unix}",
            "city": city,
            "fetched_at": datetime.now(tz=timezone.utc),
            "timestamp": timestamp_dt,
            # store the entire open-meteo response as-is
            "payload": data,
        }

        self.raw_collection.replace_one({"_id": raw_doc["_id"]}, raw_doc, upsert=True)

        # -----------------------------------------------------------------------
        # 2. Save current-only snapshot to weather_data (used by the API)
        # -----------------------------------------------------------------------
        current_doc = {
            "_id": f"{city}_{timestamp_unix}",
            "city": city,
            "timestamp": timestamp_dt,
            "processed_at": datetime.now(tz=timezone.utc),
            "data": {
                "main": {
                    "temp": current.get("temperature_2m"),
                    "feels_like": current.get("apparent_temperature"),
                    "humidity": current.get("relative_humidity_2m"),
                    "pressure": current.get("surface_pressure"),
                    "sea_level": current.get("pressure_msl"),
                    "dew_point": current.get("dew_point_2m"),
                },
                "wind": {
                    "speed": current.get("wind_speed_10m"),
                    "deg": current.get("wind_direction_10m"),
                    "gust": current.get("wind_gusts_10m"),
                },
                "clouds": {
                    "all": current.get("cloud_cover"),
                },
                "weather": [
                    {
                        "id": current.get("weather_code"),
                        "description": _weather_code_description(
                            current.get("weather_code", 0)
                        ),
                    }
                ],
                "precipitation": current.get("precipitation"),
                "rain": current.get("rain"),
                "showers": current.get("showers"),
                "snowfall": current.get("snowfall"),
                "visibility": current.get("visibility"),
                "shortwave_radiation": current.get("shortwave_radiation"),
                "uv_index": current.get("uv_index"),
                "is_day": current.get("is_day"),
            },
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "utc_offset_seconds": data.get("utc_offset_seconds"),
            "timezone": data.get("timezone"),
        }

        self.current_collection.replace_one(
            {"_id": current_doc["_id"]}, current_doc, upsert=True
        )

        temp = current.get("temperature_2m", "N/A")
        print(f"Stored: {city} @ {timestamp_dt.isoformat()} — {temp} °C", flush=True)

    def run(self):
        """Main consumer loop"""
        print(
            "Starting Weather Consumer (Open-Meteo → raw_weather + weather_data)...",
            flush=True,
        )
        for message in self.consumer:
            try:
                self.process_message(message)
                self.consumer.commit()
            except Exception as e:
                print(f"Error processing message: {e}", flush=True)
                print(f"Message value keys: {list(message.value.keys())}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weather_code_description(code: int) -> str:
    """Return a human-readable description for an Open-Meteo WMO weather code."""
    descriptions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"weather code {code}")


if __name__ == "__main__":
    consumer = WeatherConsumer()
    consumer.run()
