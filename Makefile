.PHONY: help install build up down restart clean logs test test-api test-web lint format \
        backfill sync-atlas sync-supabase coverage dev-api dev-web

help:
	@echo "install      install backend and frontend dependencies"
	@echo "dev-api      run the FastAPI service on :8000 (reload)"
	@echo "dev-web      run the Vite dev server on :5173"
	@echo "up / down    start / stop the full docker stack"
	@echo "test         run backend + frontend test suites"
	@echo "lint         ruff + black check + oxlint + tsc"
	@echo "format       black + ruff --fix"

install:
	.venv/bin/python -m pip install -r backend/requirements.txt
	cd frontend && npm ci

# --- development -----------------------------------------------------------

dev-api:
	.venv/bin/python -m uvicorn app.main:app --app-dir backend --reload --port 8000

dev-web:
	cd frontend && npm run dev

# --- containers ------------------------------------------------------------

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart: down build up

clean:
	docker compose down --rmi local
	docker image prune -f

logs:
	docker compose logs -f

# --- quality ---------------------------------------------------------------

test: test-api test-web

test-api:
	.venv/bin/python -m pytest backend/tests -q

test-web:
	cd frontend && npm test

lint:
	.venv/bin/python -m ruff check backend scripts
	.venv/bin/python -m black --check backend scripts
	cd frontend && npm run typecheck && npm run lint

format:
	.venv/bin/python -m black backend scripts
	.venv/bin/python -m ruff check --fix backend scripts

coverage:
	.venv/bin/python -m pytest backend/tests --cov=backend/app --cov-report=term-missing

# --- data ------------------------------------------------------------------

backfill:
	.venv/bin/python scripts/backfill_historical_data.py

sync-atlas:
	.venv/bin/python scripts/sync_to_atlas.py

sync-supabase:
	.venv/bin/python scripts/sync_to_supabase.py
