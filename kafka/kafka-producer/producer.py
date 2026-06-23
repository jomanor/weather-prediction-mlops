import json
import time
import os
from datetime import datetime
from kafka import KafkaProducer
import requests

class WeatherProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv('KAFKA_BROKER', 'kafka:9092'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.cities = {
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
            "Alicante": (38.3452, -0.4810)
        }
    
    def fetch_weather(self, city, coords):
        """Fetch real-time weather data from Open-Meteo API"""
        lat, lon = coords
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'wind_speed_unit': 'ms',
            'timeformat': 'unixtime',
            'latitude': lat,
            'longitude': lon,
            'current': [
                'temperature_2m',
                'relative_humidity_2m',
                'apparent_temperature',
                'is_day',
                'wind_speed_10m',
                'wind_direction_10m',
                'wind_gusts_10m',
                'precipitation',
                'rain',
                'showers',
                'weather_code',
                'cloud_cover',
                'pressure_msl',
                'surface_pressure',
                'snowfall',
                'visibility',
                'shortwave_radiation',
                'dew_point_2m',
                'uv_index'

            ],
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                data['city'] = city
                data['coordinates'] = {'lat': lat, 'lon': lon} # this is no use, the api gives the coord also
                return data
        
        except Exception as e:
            print(f"Error fetching {city}: {e}", flush=True)
        
        return None
    
    def run(self):
        """Main loop to collect and send weather data"""
        while True:
            for city, coords in self.cities.items():
                data = self.fetch_weather(city, coords)
                if data:
                    print(data)
                    self.producer.send('weather-data', value=data)
                    print(f"\n{'='*60}", flush=True)
                    print(f"Successfully sent data for {city}", flush=True)
                    print(f"Temperature: {data['current']['temperature_2m']}°C", flush=True)
                    print(f"Humidity: {data['current']['relative_humidity_2m']}%", flush=True)
                    print(f"Wind Speed: {data['current']['wind_speed_10m']} km/h", flush=True)
                    print(f"Full data structure: {json.dumps(data, indent=2)}", flush=True)
                    print(f"{'='*60}\n", flush=True)
                else:
                    print(f"✗ Failed to fetch data for {city}", flush=True)
            
            print(f"\nWaiting 120 seconds before next batch...\n", flush=True)
            time.sleep(120)

if __name__ == "__main__":
    producer = WeatherProducer()
    producer.run()