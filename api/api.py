from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MLOps Weather API :)", version="1.0.0")

# MongoDB connection
mongo_uri = os.getenv('MONGO_URL')
client = MongoClient(mongo_uri)
db = client['weather_db']
collection = db['weather_data']

@app.get("/weather/current/{city}")
def get_current_weather(city: str):
    """Get the latest weather reading for a city"""
    
    # Query
    result = collection.find_one(
        {"city": city},
        sort=[("timestamp", -1)]
    )
    
    if not result:
        raise HTTPException(status_code=404, detail=f"City {city} not found")
    
    return {
        "city": result['city'],
        "timestamp": result['timestamp'],
        "temperature": result['data']['main']['temp']
    }