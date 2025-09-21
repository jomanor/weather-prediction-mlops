import json
import time
import os
from datetime import datetime
from kafka import KafkaProducer
import requests
from dotenv import load_dotenv

load_dotenv()

class WeatherProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv('KAFKA_BROKER', 'kafka:9092'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.api_key = os.getenv('WEATHER_API_KEY')
        
        self.cities = [
            "El Ejido,ES", "Almería,ES", "Granada,ES", "Paterna,ES",
            "Madrid,ES", "Barcelona,ES", "Valencia,ES", "Sevilla,ES",
            "Zaragoza,ES", "Malaga,ES", "Murcia,ES", "Palma,ES",
            "Bilbao,ES", "Alicante,ES"
        ]
    
    def fetch_weather(self, city):
        """Fetch weather data from OpenWeatherMap"""
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching {city}: {e}", flush=True)
        return None
    
    def run(self):
        """Main loop to collect and send weather data"""
        while True:
            for city in self.cities:
                data = self.fetch_weather(city)
                if data:
                    self.producer.send('weather-data', value=data)
                    print(f"Sent data for {city}", flush=True)
            
            time.sleep(120)

if __name__ == "__main__":
    producer = WeatherProducer()
    producer.run()