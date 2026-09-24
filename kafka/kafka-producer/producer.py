import json
import os
import time

import requests
from pymongo import MongoClient

from kafka import KafkaProducer

#: Fallback stations, used verbatim when the Mongo registry is unreachable or empty.
FALLBACK_CITIES = {
    "El Ejido": (36.7756, -2.8144),
    "Almería": (36.8381, -2.4597),
    "Granada": (37.1773, -3.5986),
    "Paterna": (39.5028, -0.4408),
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3851, 2.1734),
    "Valencia": (39.4699, -0.3763),
    "Sevilla": (37.3891, -5.9845),
    "Zaragoza": (41.6488, -0.8891),
    "Malaga": (36.7213, -4.4214),
    "Murcia": (37.9922, -1.1307),
    "Palma": (39.5696, 2.6502),
    "Bilbao": (43.2630, -2.9350),
    "Alicante": (38.3452, -0.4810),
}

MONGO_DB = "weather_db"


class WeatherProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BROKER", "kafka:9092"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        self.cities = self.load_cities()

        # Pressure levels for upper-air atmospheric variables
        self.pressure_levels = [200, 500, 700, 850, 925, 1000]

    def load_cities(self):
        """Station registry from Mongo; the built-in defaults as a fallback.

        The API owns the ``cities`` collection and seeds it with the same 14
        stations. Any failure here must not stop ingestion, so it degrades to
        ``FALLBACK_CITIES`` and reports which source was used.
        """
        fallback = dict(FALLBACK_CITIES)
        uri = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
        if not uri:
            print(
                "Cities source: built-in defaults (MONGO_URI/MONGO_URL not set)",
                flush=True,
            )
            return fallback

        try:
            with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
                docs = list(
                    client[MONGO_DB].cities.find({}, {"name": 1, "latitude": 1, "longitude": 1})
                )
        except Exception as e:
            print(f"Cities source: built-in defaults (Mongo unavailable: {e})", flush=True)
            return fallback

        cities = {
            doc["name"]: (float(doc["latitude"]), float(doc["longitude"]))
            for doc in docs
            if doc.get("name")
            and doc.get("latitude") is not None
            and doc.get("longitude") is not None
        }
        if not cities:
            print("Cities source: built-in defaults (cities collection empty)", flush=True)
            return fallback

        print(f"Cities source: Mongo registry ({len(cities)} stations)", flush=True)
        return cities

    def fetch_weather(self, city, coords):
        """Fetch real-time weather data from Open-Meteo API (current + hourly + pressure levels)"""
        lat, lon = coords
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "wind_speed_unit": "ms",
            "timeformat": "unixtime",
            "latitude": lat,
            "longitude": lon,
            # Current conditions
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "is_day",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "precipitation",
                "rain",
                "showers",
                "weather_code",
                "cloud_cover",
                "pressure_msl",
                "surface_pressure",
                "snowfall",
                "visibility",
                "shortwave_radiation",
                "dew_point_2m",
                "uv_index",
            ],
            # Hourly atmospheric variables for model features
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "precipitation",
                "rain",
                "snowfall",
                "weather_code",
                "pressure_msl",
                "surface_pressure",
                "cloud_cover",
                "visibility",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "shortwave_radiation",
                "cape",  # Convective Available Potential Energy
                "wind_speed_80m",
                "wind_direction_80m",
            ],
            # Upper-air variables at pressure levels
            "pressure_level": self.pressure_levels,
            "hourly_pressure_level": [
                "geopotential_height",
                "temperature",
                "relative_humidity",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
            ],
            "forecast_hours": 24,  # Only need next 24h of hourly data
            "timezone": "auto",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                data["city"] = city
                return data
            else:
                print(
                    f"HTTP {response.status_code} for {city}: {response.text[:200]}",
                    flush=True,
                )

        except Exception as e:
            print(f"Error fetching {city}: {e}", flush=True)

        return None

    def run(self):
        """Main loop to collect and send weather data"""
        while True:
            for city, coords in self.cities.items():
                data = self.fetch_weather(city, coords)
                if data:
                    self.producer.send("weather-data", value=data)
                    current = data.get("current", {})
                    print(f"{'=' * 60}", flush=True)
                    print(f"Sent data for {city}", flush=True)
                    print(
                        f"  Temperature : {current.get('temperature_2m')} °C",
                        flush=True,
                    )
                    print(
                        f"  Humidity    : {current.get('relative_humidity_2m')} %",
                        flush=True,
                    )
                    print(
                        f"  Wind speed  : {current.get('wind_speed_10m')} m/s",
                        flush=True,
                    )
                    print(
                        f"  Hourly rows : {len(data.get('hourly', {}).get('time', []))}",
                        flush=True,
                    )
                    print(f"{'=' * 60}\n", flush=True)
                else:
                    print(f"✗ Failed to fetch data for {city}", flush=True)

            print("\nWaiting 120 seconds before next batch...\n", flush=True)
            time.sleep(120)


if __name__ == "__main__":
    producer = WeatherProducer()
    producer.run()
