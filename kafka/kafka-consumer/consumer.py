import json
import os
from datetime import datetime
from kafka import KafkaConsumer
from pymongo import MongoClient

class WeatherConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'weather-data',
            bootstrap_servers=os.getenv('KAFKA_BROKER', 'kafka:9092'),
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            group_id='weather-consumer-group',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        mongo_url = os.getenv('MONGO_URL')
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client['weather_db']
        self.collection = self.db['weather_data']
    
    def map_weather_code(self, code):
        """Map Open-Meteo weather codes to OpenWeatherMap-like format"""
        weather_mapping = {
            0: {'main': 'Clear', 'description': 'clear sky', 'icon': '01d'},
            1: {'main': 'Clouds', 'description': 'mainly clear', 'icon': '02d'},
            2: {'main': 'Clouds', 'description': 'partly cloudy', 'icon': '03d'},
            3: {'main': 'Clouds', 'description': 'overcast clouds', 'icon': '04d'},
            45: {'main': 'Fog', 'description': 'fog', 'icon': '50d'},
            48: {'main': 'Fog', 'description': 'depositing rime fog', 'icon': '50d'},
            51: {'main': 'Drizzle', 'description': 'light drizzle', 'icon': '09d'},
            53: {'main': 'Drizzle', 'description': 'moderate drizzle', 'icon': '09d'},
            55: {'main': 'Drizzle', 'description': 'dense drizzle', 'icon': '09d'},
            61: {'main': 'Rain', 'description': 'slight rain', 'icon': '10d'},
            63: {'main': 'Rain', 'description': 'moderate rain', 'icon': '10d'},
            65: {'main': 'Rain', 'description': 'heavy rain', 'icon': '10d'},
            71: {'main': 'Snow', 'description': 'slight snow', 'icon': '13d'},
            73: {'main': 'Snow', 'description': 'moderate snow', 'icon': '13d'},
            75: {'main': 'Snow', 'description': 'heavy snow', 'icon': '13d'},
            95: {'main': 'Thunderstorm', 'description': 'thunderstorm', 'icon': '11d'},
        }
        return weather_mapping.get(code, {'main': 'Unknown', 'description': 'unknown', 'icon': '01d'})
        
    def process_message(self, message):
        """Process and store weather data from Open-Meteo in OpenWeatherMap-compatible format"""
        data = message.value
        
        city = data.get('city', 'unknown')
        coords = data.get('coordinates', {})
        current = data.get('current', {})
        hourly = data.get('hourly', {})
        
        current_time = current.get('time', '')
        try:
            timestamp_dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            timestamp_unix = int(timestamp_dt.timestamp())
        except:
            timestamp_dt = datetime.now()
            timestamp_unix = int(timestamp_dt.timestamp())
        
        visibility = None
        if hourly.get('time') and hourly.get('visibility'):
            try:
                hourly_times = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in hourly['time']]
                current_hour_index = min(range(len(hourly_times)), 
                                       key=lambda i: abs((hourly_times[i] - timestamp_dt).total_seconds()))
                visibility = hourly['visibility'][current_hour_index]
            except:
                visibility = None
        
        doc_id = f"{city}_{timestamp_unix}"
        
        weather_code = current.get('weather_code', 0)
        weather_info = self.map_weather_code(weather_code)
        
        data_obj = {
            "coord": {
                "lon": coords.get('lon'),
                "lat": coords.get('lat')
            },
            "weather": [{
                "id": weather_code,
                "main": weather_info['main'],
                "description": weather_info['description'],
                "icon": weather_info['icon']
            }],
            "base": "open-meteo",
            "main": {
                "temp": current.get('temperature_2m'),
                "feels_like": current.get('apparent_temperature'),
                "pressure": current.get('surface_pressure'),
                "humidity": current.get('relative_humidity_2m'),
                "sea_level": current.get('pressure_msl')
            },
            "visibility": visibility,
            "wind": {
                "speed": current.get('wind_speed_10m'),
                "deg": current.get('wind_direction_10m'),
                "gust": current.get('wind_gusts_10m')
            },
            "rain": current.get('rain'),
            "showers": current.get('showers'),
            "snow": current.get('snowfall'),
            "precipitation": current.get('precipitation'),
            "clouds": {
                "all": current.get('cloud_cover')
            },
            "is_day": current.get('is_day'),
            "dt": timestamp_unix,
            "sys": {
                "country": "ES"
            },
            "timezone": data.get('utc_offset_seconds'),
            "name": city,
            "cod": 200,
        }
        
        document = {
            "_id": doc_id,
            "city": city,
            "data": data_obj,
            "timestamp": timestamp_dt,
            "processed_at": datetime.now()
        }
        
        self.collection.replace_one(
            {"_id": doc_id},
            document,
            upsert=True
        )
        
        temp = current.get('temperature_2m', 'N/A')
        print(f"Processed: {city} at {timestamp_unix} - Temp: {temp}°C", flush=True)
        
    def run(self):
        """Main consumer loop"""
        print("Starting Weather Consumer for Open-Meteo data...", flush=True)
        for message in self.consumer:
            try:
                self.process_message(message)
                self.consumer.commit()
            except Exception as e:
                print(f"Error processing message: {e}", flush=True)
                print(f"Message value: {message.value}", flush=True)

if __name__ == "__main__":
    consumer = WeatherConsumer()
    consumer.run()