.PHONY: build up down restart clean logs test lint format backfill sync-atlas sync-supabase coverage

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart: down build up

clean:
	docker compose down --rmi local
	docker rmi mongo:6.0 apache/kafka:4.1.0 apache/spark
	docker image prune -f

logs:
	docker compose logs

test:
	pytest tests/ -v

lint:
	ruff check . && black --check .

format:
	black . && ruff check --fix .

backfill:
	python scripts/backfill_historical_data.py

sync-atlas:
	python scripts/sync_to_atlas.py

sync-supabase:
	python scripts/sync_to_supabase.py

coverage:
	pytest tests/ --cov=kafka/kafka-consumer --cov=kafka/kafka-producer --cov=spark/spark-jobs --cov=scripts --cov=api --cov-report=html