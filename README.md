# MeteoML — Weather Prediction & MLOps Platform

An end-to-end weather forecasting stack: real-time ingestion, atmospheric
feature engineering, distributed model training, and an honest benchmark of the
resulting model against the official AEMET forecast.

The platform is built around one rule: **the UI never invents data.** Every
number on screen comes from the database, from AEMET, or from a documented
physical formula. When a source is unavailable, the field is `null` and the
interface says so.

---

## Architecture

```
                       [ Open-Meteo API ]
                               │
                               ▼
                    [ Kafka weather producer ]
                               │
                        ( Kafka broker )
                               │
                               ▼
                    [ Kafka weather consumer ]
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        [ MongoDB: raw_weather ]   [ Historical backfill ]
                  │
                  ▼
        [ Spark feature pipeline ]
        · Bolton specific humidity (q)
        · Cartesian wind vectors (u, v)
        · CAPE and pressure levels (200–1000 hPa)
        · Cyclical time, rolling precipitation
                  │
                  ▼
        [ Spark ML — Gradient Boosted Trees ]
                  │
                  ├──────────────► [ MLflow tracking + registry ]
                  ▼
        [ Batch inference job ]
                  │
                  ▼
        [ MongoDB: weather_predictions ]
                  │
                  ▼
        [ FastAPI backend ] ◄──── [ AEMET OpenData ]
                  │
                  ▼
        [ React frontend (Vite + TypeScript) ]
                  │
                  ▼
        [ Netlify edge  ·  nginx container ]
```

---

## Repository layout

```
weather-prediction-mlops/
├── backend/                    # FastAPI service
│   ├── app/
│   │   ├── main.py             # app factory, CORS, lifespan
│   │   ├── core/               # settings, logging, coercion helpers
│   │   ├── db/                 # Mongo client lifecycle + indexes
│   │   ├── schemas/            # Pydantic request/response models
│   │   ├── repositories/       # one module per collection, single mapper
│   │   ├── services/           # AEMET client, benchmark maths, window stats
│   │   └── routers/            # health, weather, predictions, benchmark, models
│   ├── tests/                  # 59 offline tests (fakes, no live Mongo)
│   └── Dockerfile
├── frontend/                   # React + TypeScript + Tailwind (Vite)
│   ├── src/
│   │   ├── api/                # typed client + zod schemas for the contract
│   │   ├── app/                # shell, navigation, preferences
│   │   ├── components/         # ui primitives, charts, map, shell controls
│   │   ├── features/           # overview, station, benchmark, models
│   │   └── lib/                # formatting, units, derived physics, theme
│   ├── nginx.conf              # SPA fallback + /api reverse proxy
│   └── Dockerfile
├── kafka/                      # producer + consumer
├── spark/                      # feature engineering, training, inference, scheduler
├── scripts/                    # backfill, Atlas sync, Supabase sync
├── tests/                      # pipeline unit + integration tests
├── docs/api-contract.md        # single source of truth for backend ⇄ frontend
├── docker-compose.yml          # full local stack
├── netlify.toml                # production build + edge API proxy
└── Makefile
```

---

## The benchmark

The headline feature is a like-for-like comparison on the same hours:

| Series    | Source                                                    |
| --------- | --------------------------------------------------------- |
| Observado | Open-Meteo observations stored in MongoDB                  |
| Spark GBT | The platform's own model, served from `weather_predictions` |
| AEMET     | AEMET OpenData hourly municipal forecast                   |

`GET /api/benchmark/{city}` joins the three on timestamp and returns the series
plus MAE, RMSE and bias for each model. If AEMET is unreachable or no key is
configured, the response sets `aemet.available = false` with an explanatory
error and the UI renders that state — it never substitutes a stand-in curve.

The contract is defined once in [`docs/api-contract.md`](docs/api-contract.md)
and enforced on both sides: Pydantic models in the backend, zod schemas in the
frontend. A drifting response fails loudly in development instead of rendering
`undefined`.

---

## Quick start

### Full stack (Docker)

```bash
cp .env.example .env          # fill in MONGO_* and AEMET_API_KEY
make up
```

| Service        | URL                     |
| -------------- | ----------------------- |
| Frontend       | http://localhost:8080   |
| API            | http://localhost:8000   |
| API docs       | http://localhost:8000/docs |
| MLflow         | http://localhost:5000   |
| Spark master   | http://localhost:8080   |

### Local development

```bash
make install                  # backend deps + npm ci

make dev-api                  # FastAPI on :8000 (reload)
make dev-web                  # Vite on :5173, proxies /api → :8000
```

The Vite dev server proxies `/api` to `http://localhost:8000`, so the frontend
always uses the same relative base URL it uses in production.

---

## Configuration

| Variable          | Used by  | Default                 | Notes                                        |
| ----------------- | -------- | ----------------------- | -------------------------------------------- |
| `MONGO_URI`       | backend, consumer, spark | `mongodb://localhost:27017` | `MONGO_URL` accepted for compatibility |
| `MONGO_DB`        | backend  | `weather_db`            |                                              |
| `CORS_ORIGINS`    | backend  | `http://localhost:5173` | comma separated                              |
| `AEMET_API_KEY`   | backend  | —                       | required for the benchmark                   |
| `VITE_API_BASE_URL` | frontend | `/api`                | only set this to point at another host       |

---

## Testing

```bash
make test          # backend (59) + frontend (29)
make lint          # ruff + black + oxlint + tsc
make coverage      # backend coverage report
```

Backend tests are fully offline: routers run against dependency-overridden
fakes and the AEMET client runs against `httpx.MockTransport`, including a real
ISO-8859-15 fixture with missing hourly values.

---

## Deployment

- **Frontend** — Netlify builds `frontend/` and publishes `frontend/dist`.
  `/api/*` is proxied at the edge to the FastAPI service, so the browser only
  ever makes same-origin requests. Hashed assets are cached immutably.
- **Backend** — any container host; the image is `python:3.13-slim`, runs as a
  non-root user, and reads all configuration from the environment.
- **Database** — MongoDB Atlas. `scripts/sync_to_atlas.py` mirrors the local
  pipeline output to the cloud cluster.
