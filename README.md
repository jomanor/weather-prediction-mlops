# End-to-End Weather Prediction & MLOps Platform

An enterprise-grade, distributed MLOps architecture designed for real-time weather data ingestion, atmospheric physics feature engineering, distributed Machine Learning training (Apache Spark GBT), experiment tracking & model registry (MLflow), and real-time prediction benchmarking against official meteorological models (AEMET).

---

## System Architecture

```
                                  [ Open-Meteo API ]
                                          │
                                          ▼
                               [ Kafka Weather Producer ]
                                          │
                                   (Kafka Broker)
                                          │
                                          ▼
                               [ Kafka Weather Consumer ]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
          [ MongoDB / Supabase ]                     [ Historical Backfill ]
          (raw_weather, features)                                │
                   │                                             ▼
                   ├─────────────────────────────────────────────┘
                   ▼
     [ Apache Spark Feature Pipeline ]
     - Bolton Specific Humidity (q)
     - Cartesian Wind Vectors (u, v)
     - CAPE & Pressure Levels (200-1000 hPa)
     - Cyclical Time & Rolling Precip Sum
                   │
                   ▼
       [ Apache Spark ML (GBT) ] ────► [ MLflow Model Registry ]
                   │                             │
                   ▼                             ▼
       [ Batch Inference Engine ] ◄──────────────┘
                   │
                   ▼
       [ MongoDB / Supabase DB ]
         (weather_predictions)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  [ FastAPI Service ]  [ Streamlit Dashboard ]
    (Port 8000)          (Port 8501)
         │                   │
         └─────────┬─────────┘
                   ▼
         [ Netlify Production Deployment ]
```

---

## Repository Structure

```
weather-prediction-mlops/
├── api/                        # FastAPI REST microservice
│   ├── api.py                  # Endpoints (/predictions, /weather/current, /stats)
│   ├── models.py               # Pydantic data validation schemas
│   └── Dockerfile              # Container spec for API service
├── dashboard/                  # Streamlit Web Dashboard
│   ├── app.py                  # Main entrypoint & UI design system
│   ├── pages/                  # Multipage dashboard sections
│   │   ├── 1_comparacion.py    # Live model vs AEMET vs Observed chart
│   │   ├── 2_historico.py      # Historical error & metrics log
│   │   ├── 3_modelo.py         # MLflow registry audit & model info
│   │   └── 4_observado.py      # Live sensor telemetry & atmospheric charts
│   ├── components/             # Reusable UI components & clients
│   │   ├── aemet_client.py     # AEMET OpenData API client with 1h TTL cache
│   │   ├── mongo_client.py     # MongoDB query helper
│   │   ├── supabase_client.py  # Supabase REST client for cloud tier
│   │   ├── charts.py           # Plotly visualization theme
│   │   ├── i18n.py             # Multilingual localization (ES / EN)
│   │   └── theme.py            # Glassmorphism dark UI theme
│   ├── requirements.txt        # Dashboard dependencies
│   └── Dockerfile              # Streamlit container spec
├── kafka/                      # Event streaming pipeline
│   ├── kafka-producer/         # Ingests 14 Iberian cities + 6 pressure levels
│   └── kafka-consumer/         # Writes raw payloads & upserts current status
├── spark/                      # Distributed Spark pipeline & scheduler
│   ├── spark-jobs/
│   │   ├── batch_processing.py # Feature engineering (deduplicated by city+time)
│   │   ├── ml_training.py      # GBTRegressor training & MLflow tracking
│   │   └── inference.py        # Model loading (MLflow Registry / GridFS fallback)
│   ├── scheduler.py            # Cron runner for batch & inference jobs
│   └── spark_config.py         # Spark session & feature config
├── scripts/
│   ├── backfill_historical_data.py # Idempotent historical data loader
│   ├── sync_to_atlas.py        # Synchronizes MongoDB to MongoDB Atlas Cloud
│   └── sync_to_supabase.py     # Synchronizes MongoDB to Supabase PostgreSQL
├── tests/                      # Unit & integration test suite
│   ├── unit/                   # 54 unit tests
│   └── integration/            # Ingestion pipeline integration tests
├── docker-compose.yml          # Full multi-container stack definition
├── Makefile                    # Developer commands (make up, test, lint, format)
├── netlify.toml                # Netlify deployment & serverless edge proxy config
└── pyproject.toml              # Tool configs (ruff, black, pytest)
```

---

## Cloud Hosting & Resource Management (Netlify + MongoDB Atlas)

The platform is designed to operate seamlessly within managed cloud tiers without requiring credit card verification:

### 1. Web Hosting & Edge Proxy on Netlify
- **Architecture**:
  - The Streamlit Dashboard / frontend interface is deployed to Netlify edge infrastructure.
  - API routes (`/api/*`) are configured via `netlify.toml` edge proxy rules to route to the backend REST service, resolving cross-origin requests cleanly.
  - Static visual assets and chart components enforce edge caching (`Cache-Control: public, max-age=3600`).

### 2. Database Integration on MongoDB Atlas (Free M0 Tier)
- **Data Layer**:
  - Official **MongoDB Atlas Free M0 Cluster** providing 512 MB cloud document storage (no credit card required).
  - Sync utility `scripts/sync_to_atlas.py` pushes local MongoDB pipeline outputs (`raw_weather`, `weather_features`, `weather_predictions`) directly to Atlas over TLS wire protocol.


### 3. Data Lifecycle & Storage Optimization Strategy
- **Automated Data Pruning**: A rolling retention policy purges high-frequency raw observations older than 30 days while preserving daily/hourly climate summaries.
- **Inference Rate Control**: Batch inference runs once per hour across all 14 monitored municipalities, maintaining predictable dataset growth (~336 rows/day).
- **Deduplication**: Pipeline jobs enforce strict `dropDuplicates(["city", "timestamp"])` prior to database writes.

---

## Deployment & Local Execution

### Local Environment Setup
To launch the full containerized stack locally:
```bash
docker compose up -d
```

### Key Service Endpoints
- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI REST API**: `http://localhost:8000` (Swagger UI at `http://localhost:8000/docs`)
- **MLflow Tracking Server**: `http://localhost:5000`
- **Spark Master UI**: `http://localhost:8080`

### Pipeline Execution Commands
- Run historical backfill:
  ```bash
  make backfill
  ```
- Execute test suite:
  ```bash
  make test
  ```
- Lint codebase:
  ```bash
  make lint
  ```
