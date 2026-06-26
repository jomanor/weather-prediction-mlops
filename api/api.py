from fastapi import FastAPI, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
from typing import List, Optional
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging

# Pydantic models
from models import (
    CurrentWeatherResponse,
    WeatherStatsResponse,
    HistoricalWeatherResponse,
    WeatherPredictionResponse,
    LatestPredictionsResponse,
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# TODO Maybe get rid off of global variables and use app.state in lifespan function
motor_client = None
db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global motor_client, db
    
    mongo_uri = os.getenv('MONGO_URL')
    motor_client = AsyncIOMotorClient(mongo_uri)
    
    try:
        await motor_client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    
    db = motor_client['weather_db']
    
    await db.weather_data.create_index([("city", 1), ("timestamp", -1)])
    await db.weather_predictions.create_index([("city", 1), ("prediction_timestamp", -1)])
    logger.info("Indexes created")
    
    yield 
    
    logger.info("Shutting down...")
    motor_client.close()


app = FastAPI(
    title="MLOps Weather API :)",
    version="2.0.0",
    description="Async Weather API!!",
    lifespan=lifespan
)

@app.get("/")
async def root():
    
    return {
        "status": "healthy",
        "service": "Weather API",
        "version": "2.0.0"
    }

@app.get("/weather/current/{city}", response_model=CurrentWeatherResponse)
async def get_current_weather(city: str):

    try:
        result = await db.weather_data.find_one(
            {"city": city},
            sort=[("timestamp", -1)]
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")

        return CurrentWeatherResponse(
            city=result['city'],
            temperature=result.get('temperature', result.get('data', {}).get('main', {}).get('temp')),
            feels_like=result.get('feels_like', result.get('data', {}).get('main', {}).get('feels_like')),
            humidity=result.get('humidity', result.get('data', {}).get('main', {}).get('humidity')),
            pressure=result.get('pressure', result.get('data', {}).get('main', {}).get('pressure')),
            wind_speed=result.get('wind_speed', result.get('data', {}).get('wind', {}).get('speed')),
            description=result.get('description', 'N/A'),
            timestamp=result['timestamp'],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching current weather: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/weather/cities", response_model=List[str])
async def get_all_cities():
    
    try:
        cities = await db.weather_data.distinct("city")
        return sorted(cities)
    except Exception as e:
        logger.error(f"Error fetching cities: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch cities")

@app.get("/weather/current", response_model=List[CurrentWeatherResponse])
async def get_all_current_weather():

    try:
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$city",
                "latest": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$latest"}}
        ]

        cursor = db.weather_data.aggregate(pipeline)
        results = []

        async for doc in cursor:
            results.append(CurrentWeatherResponse(
                city=doc['city'],
                temperature=doc.get('temperature', doc.get('data', {}).get('main', {}).get('temp')),
                feels_like=doc.get('feels_like', doc.get('data', {}).get('main', {}).get('feels_like')),
                humidity=doc.get('humidity', doc.get('data', {}).get('main', {}).get('humidity')),
                pressure=doc.get('pressure', doc.get('data', {}).get('main', {}).get('pressure')),
                wind_speed=doc.get('wind_speed', doc.get('data', {}).get('wind', {}).get('speed')),
                description=doc.get('description', 'N/A'),
                timestamp=doc['timestamp'],
            ))

        return results
    except Exception as e:
        logger.error(f"Error fetching all current weather: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")

@app.get("/weather/historical/{city}")
async def get_historical_weather(
    city: str,
    hours: int = Query(default=24, ge=1, le=168, description="Hours of historical data (max 168 = 1 week)"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of records")
):
    
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        query = {
            "city": city,
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }
        
        cursor = db.weather_data.find(query).sort("timestamp", -1).limit(limit)

        data = []
        async for doc in cursor:
            data.append(CurrentWeatherResponse(
                city=doc['city'],
                temperature=doc.get('temperature', doc.get('data', {}).get('main', {}).get('temp')),
                feels_like=doc.get('feels_like', doc.get('data', {}).get('main', {}).get('feels_like')),
                humidity=doc.get('humidity', doc.get('data', {}).get('main', {}).get('humidity')),
                pressure=doc.get('pressure', doc.get('data', {}).get('main', {}).get('pressure')),
                wind_speed=doc.get('wind_speed', doc.get('data', {}).get('wind', {}).get('speed')),
                description=doc.get('description', 'N/A'),
                timestamp=doc['timestamp'],
            ))
        
        if not data:
            raise HTTPException(status_code=404, detail=f"No historical data found for city '{city}'")
        
        return {
            "city": city,
            "period_hours": hours,
            "count": len(data),
            "data": data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical weather: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch historical data")

@app.get("/weather/stats/{city}", response_model=WeatherStatsResponse)
async def get_weather_statistics(
    city: str,
    hours: int = Query(default=24, ge=1, le=168, description="Period for statistics")
):

    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        pipeline = [
            {
                "$match": {
                    "city": city,
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            },
            {
                "$group": {
                    "_id": "$city",
                    "avg_temp": {"$avg": "$data.main.temp"},
                    "min_temp": {"$min": "$data.main.temp"},
                    "max_temp": {"$max": "$data.main.temp"},
                    "avg_humidity": {"$avg": "$data.main.humidity"},
                    "avg_pressure": {"$avg": "$data.main.pressure"},
                    "count": {"$sum": 1},
                    "start_time": {"$min": "$timestamp"},
                    "end_time": {"$max": "$timestamp"}
                }
            }
        ]
        
        cursor = db.weather_data.aggregate(pipeline)
        result = await cursor.to_list(1)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"No data found for city '{city}'")
        
        stats = result[0]
        return WeatherStatsResponse(
            city=stats['_id'],
            period_hours=hours,
            avg_temperature=round(stats['avg_temp'], 2),
            min_temperature=round(stats['min_temp'], 2),
            max_temperature=round(stats['max_temp'], 2),
            avg_humidity=round(stats['avg_humidity'], 2),
            avg_pressure=round(stats['avg_pressure'], 2),
            total_records=stats['count'],
            start_time=stats['start_time'],
            end_time=stats['end_time']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate statistics")

@app.get("/weather/compare")
async def compare_cities(
    cities: str = Query(description="Comma-separated list of cities (e.g., 'Madrid,Barcelona,Valencia')"),
    hours: int = Query(default=24, ge=1, le=168, description="Period for comparison")
):

    try:
        city_list = [c.strip() for c in cities.split(",")]
        
        if len(city_list) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 cities to compare")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        comparisons = []
        
        for city in city_list:
            pipeline = [
                {
                    "$match": {
                        "city": city,
                        "timestamp": {"$gte": start_time, "$lte": end_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$city",
                        "avg_temp": {"$avg": "$data.main.temp"},
                        "avg_humidity": {"$avg": "$data.main.humidity"},
                        "avg_pressure": {"$avg": "$data.main.pressure"},
                        "avg_wind_speed": {"$avg": "$data.wind.speed"},
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            cursor = db.weather_data.aggregate(pipeline)
            result = await cursor.to_list(1)
            
            if result:
                stats = result[0]
                comparisons.append({
                    "city": stats['_id'],
                    "avg_temperature": round(stats['avg_temp'], 2),
                    "avg_humidity": round(stats['avg_humidity'], 2),
                    "avg_pressure": round(stats['avg_pressure'], 2),
                    "avg_wind_speed": round(stats['avg_wind_speed'], 2),
                    "data_points": stats['count']
                })
        
        if not comparisons:
            raise HTTPException(status_code=404, detail="No data found for specified cities")
        
        comparisons.sort(key=lambda x: x['avg_temperature'], reverse=True)
        
        return {
            "period_hours": hours,
            "cities_analyzed": len(comparisons),
            "comparison": comparisons
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing cities: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare cities")

# ---------------------------------------------------------------------------
# Predictions endpoints
# ---------------------------------------------------------------------------

def _doc_to_prediction(doc: dict) -> WeatherPredictionResponse:
    """Convert a MongoDB weather_predictions document to the response model."""
    return WeatherPredictionResponse(
        city=doc['city'],
        source_timestamp=doc['source_timestamp'],
        prediction_timestamp=doc['prediction_timestamp'],
        predicted_temperature=doc.get('predicted_temperature'),
        predicted_rain=doc.get('predicted_rain'),
        observed_temperature=doc.get('observed_temperature'),
        horizon_hours=doc.get('horizon_hours', 1),
        temp_model_name=doc.get('temp_model_name'),
        temp_model_version=doc.get('temp_model_version'),
        rain_model_name=doc.get('rain_model_name'),
        rain_model_version=doc.get('rain_model_version'),
    )


@app.get("/predictions/latest", response_model=LatestPredictionsResponse)
async def get_latest_predictions():
    """Return the single most-recent prediction for every city."""
    try:
        pipeline = [
            {"$sort": {"prediction_timestamp": -1}},
            {"$group": {
                "_id": "$city",
                "latest": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$latest"}},
            {"$sort": {"city": 1}},
        ]

        cursor = db.weather_predictions.aggregate(pipeline)
        predictions = [_doc_to_prediction(doc) async for doc in cursor]

        return LatestPredictionsResponse(count=len(predictions), predictions=predictions)
    except Exception as e:
        logger.error(f"Error fetching latest predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch latest predictions")


@app.get("/predictions/{city}", response_model=List[WeatherPredictionResponse])
async def get_predictions_for_city(
    city: str,
    limit: int = Query(default=24, ge=1, le=500, description="Maximum number of records to return"),
):
    """Return the most recent model predictions for *city*, newest first."""
    try:
        cursor = (
            db.weather_predictions
            .find({"city": city})
            .sort("prediction_timestamp", -1)
            .limit(limit)
        )
        results = [_doc_to_prediction(doc) async for doc in cursor]

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for city '{city}'. "
                       "Run the inference job (ml_training.py → inference.py) first."
            )

        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching predictions for {city}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch predictions")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)