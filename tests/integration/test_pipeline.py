"""
test_pipeline.py
================
Integration tests for the ingestion and processing pipeline.
Tests end-to-end flow from Kafka consumer message processing through MongoDB collections.
"""

import os
import sys
from unittest.mock import patch

import mongomock

# Ensure kafka-consumer and tests fixtures are discoverable
CONSUMER_DIR = os.path.join(os.path.dirname(__file__), "../../kafka/kafka-consumer")
TESTS_DIR = os.path.join(os.path.dirname(__file__), "..")
for d in (CONSUMER_DIR, TESTS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import consumer  # noqa: E402
from conftest import make_open_meteo_message  # noqa: E402


class TestIngestionPipelineIntegration:
    def test_consumer_full_flow_to_mock_mongo(self):
        """Verify that a raw Open-Meteo Kafka payload is parsed and correctly stored in both
        raw_weather and weather_data collections with deduplication idempotency.
        """
        mongo_client = mongomock.MongoClient()

        with patch("consumer.KafkaConsumer"):
            with patch("consumer.MongoClient", return_value=mongo_client):
                with patch.dict(
                    os.environ,
                    {"MONGO_URI": "mongodb://localhost:27017", "KAFKA_BROKER": "localhost:9092"},
                ):
                    c = consumer.WeatherConsumer()
                    c.raw_collection = mongo_client["weather_db"]["raw_weather"]
                    c.current_collection = mongo_client["weather_db"]["weather_data"]

                    from unittest.mock import MagicMock

                    # 1. Process first message for Madrid
                    msg_1 = make_open_meteo_message("Madrid", 1_750_000_000)
                    c.process_message(MagicMock(value=msg_1))

                    assert c.raw_collection.count_documents({}) == 1
                    assert c.current_collection.count_documents({}) == 1

                    raw_doc = c.raw_collection.find_one({"_id": "Madrid_1750000000"})
                    assert raw_doc is not None
                    assert raw_doc["city"] == "Madrid"
                    assert raw_doc["payload"]["current"]["temperature_2m"] == 22.5

                    curr_doc = c.current_collection.find_one({"city": "Madrid"})
                    assert curr_doc is not None
                    assert curr_doc["data"]["main"]["temp"] == 22.5
                    assert curr_doc["data"]["main"]["feels_like"] == 21.0

                    # 2. Process exact duplicate message -> Count should remain 1 (idempotency)
                    c.process_message(MagicMock(value=msg_1))
                    assert c.raw_collection.count_documents({}) == 1
                    assert c.current_collection.count_documents({}) == 1

                    # 3. New message with new timestamp for Madrid -> Inserts second document
                    msg_2 = make_open_meteo_message("Madrid", 1_750_003_600)
                    msg_2["current"]["temperature_2m"] = 26.0
                    c.process_message(MagicMock(value=msg_2))

                    assert c.raw_collection.count_documents({}) == 2
                    assert c.current_collection.count_documents({}) == 2

                    updated_curr = c.current_collection.find_one({"_id": "Madrid_1750003600"})
                    assert updated_curr["data"]["main"]["temp"] == 26.0
