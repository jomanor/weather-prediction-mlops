import json
import os
from datetime import datetime
from kafka import KafkaConsumer
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

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
        
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://mongodb:27017')
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client['weather_db']
        self.collection = self.db['weather_data']
        
    def process_message(self, message):
        """Process and store weather data"""
        data = message.value
        
        city = data.get('name', 'unknown')
        timestamp = data.get('dt', 0)
        doc_id = f"{city}_{timestamp}"
        
        document = {
            "_id": doc_id,
            "city": city,
            "data": data,
            "timestamp": datetime.fromtimestamp(timestamp),
            "processed_at": datetime.now()
        }
        
        self.collection.replace_one(
            {"_id": doc_id},
            document,
            upsert=True
        )
        
        print(f"Processed: {city} at {timestamp}", flush=True)
        
    def run(self):
        """Main consumer loop"""
        for message in self.consumer:
            try:
                self.process_message(message)
                self.consumer.commit()
            except Exception as e:
                print(f"Error processing message: {e}", flush=True)

if __name__ == "__main__":
    consumer = WeatherConsumer()
    consumer.run()