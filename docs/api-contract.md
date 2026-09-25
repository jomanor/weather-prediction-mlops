# API contract v2

Single source of truth for the `backend/` service and the `frontend/` app.
Both sides must match this document. Changing it means changing both sides
in the same PR.

Base path: `/api`
Content type: `application/json`
Timestamps: ISO-8601 UTC strings (`2026-09-23T14:00:00Z`).
Nullable fields are always present in the payload, `null` when unknown —
never omitted.

## Health

`GET /api/health`

```json
{ "status": "ok", "service": "weather-api", "version": "2.0.0", "time": "2026-09-23T14:00:00Z" }
```

## Weather

### `GET /api/cities`

```json
["Alicante", "Barcelona", "Madrid"]
```

### `GET /api/weather/current`

Latest observation per city.

```json
{ "count": 2, "stations": [CurrentWeather, ...] }
```

### `GET /api/weather/current/{city}`

`CurrentWeather` or `404`.

### `GET /api/weather/history/{city}?hours=24&limit=200`

```json
{ "city": "Madrid", "hours": 24, "count": 24, "points": [WeatherPoint, ...] }
```

### `GET /api/weather/stats/{city}?hours=24`

```json
{
  "city": "Madrid", "hours": 24, "count": 24,
  "temperature": { "avg": 21.4, "min": 14.2, "max": 27.9 },
  "humidity": { "avg": 61.0, "min": 40.0, "max": 82.0 },
  "pressure": { "avg": 1014.2, "min": 1011.0, "max": 1017.0 },
  "wind_speed": { "avg": 11.3, "min": 3.0, "max": 22.4 },
  "precipitation_total": 0.4,
  "start": "2026-09-22T14:00:00Z", "end": "2026-09-23T13:00:00Z"
}
```

### `GET /api/weather/quality?days=7`

Data-quality meter over `weather_features` (the model's actual input), one
entry per city in the window `[now - days, now]`. `days` is an integer 1..30,
default 7. Cached 300 s with an ETag/`Cache-Control` like the other read
routes. `completeness = observed_hours / (days * 24)` clamped to 1.0;
`max_gap_hours` is the largest separation between consecutive observed hours
(0 when there are fewer than two points); `null_rate` is the fraction of
observed rows with a null `temperature` (null when there are no rows);
`age_hours = now - last_observed_at`.

`status` grades completeness (≥ 0.95 / ≥ 0.80) and age (≤ 12 h / ≤ 18 h)
independently; the worst of the two wins and a missing last observation is
`bad`. Exactly those inclusivity edges apply (`completeness == 0.95` and
`age_hours == 12.0` are `ok`; `0.80`/`18.0` are `warn`). The age gates are
calibrated to the ~6 h feature-rebuild cycle (~2x ok, ~3x warn — one missed
rebuild); the target-drop that previously forced a healthy pipeline to land
~6 h stale was fixed in Batch 4, so the gates are honest, not loosened.

The response always carries one entry per canonical station (the 14 in
`app/core/cities.py`), so a city whose feature rows stopped arriving is
reported `bad` with `observed_hours: 0` instead of disappearing. Duplicate
`(city, timestamp)` rows are counted once.

```json
{
  "generated_at": "2026-09-25T12:00:00Z",
  "days": 7,
  "cities": [
    {
      "city": "Madrid",
      "expected_hours": 168,
      "observed_hours": 165,
      "completeness": 0.982,
      "max_gap_hours": 3.0,
      "null_rate": 0.012,
      "last_observed_at": "2026-09-25T11:00:00Z",
      "age_hours": 1.0,
      "status": "ok"
    }
  ]
}
```

## Predictions (real Spark GBT output)

### `GET /api/predictions/latest?horizon=`

`horizon` is an optional integer 1..48. Omitted, the response carries one row
per `(city, horizon_hours)` (cities × horizons) — the newest
`prediction_timestamp` for each pair. Provided, it is filtered to that
horizon, one row per city. `generated_at` is the max `prediction_timestamp`
across the returned rows (null when empty).

```json
{ "count": 70, "generated_at": "2026-09-23T14:00:00Z", "predictions": [Prediction, ...] }
```

### `GET /api/predictions/{city}?limit=48&horizon=`

`Prediction[]` or `404` when the city has no predictions. `limit` is 1..500
(default 48) and keeps the latest window; `horizon` is an optional 1..48
filter. The response is chronological, oldest → newest.

## Benchmark: model vs AEMET vs observed

### `GET /api/benchmark/{city}?hours=24`

```json
{
  "city": "Madrid",
  "hours": 24,
  "generated_at": "2026-09-23T14:00:00Z",
  "aemet": { "available": true, "error": null, "issued_at": "2026-09-23T06:00:00Z" },
  "series": [
    {
      "timestamp": "2026-09-23T14:00:00Z",
      "observed": 24.1,
      "model": 23.6,
      "aemet": 24.8,
      "residual_model": -0.5,
      "residual_aemet": 0.7
    }
  ],
  "metrics": {
    "model": { "mae": 1.12, "rmse": 1.44, "bias": -0.08, "n": 24 },
    "aemet": { "mae": 1.31, "rmse": 1.70, "bias": 0.22, "n": 24 }
  }
}
```

`aemet.available` is `false` and `aemet.error` explains why when the AEMET
OpenData API is unreachable or no key is configured. The `series` is still
returned with `aemet: null`. The UI must render that state honestly — never
substitute fabricated numbers.

### `GET /api/benchmark`

Per-city summary, no series.

```json
{
  "generated_at": "...",
  "aemet_configured": true,
  "cities": [
    { "city": "Madrid", "n": 24, "model": { "mae": 1.12, "rmse": 1.44, "bias": -0.08, "n": 24 },
      "aemet": { "mae": 1.31, "rmse": 1.70, "bias": 0.22, "n": 24 } }
  ]
}
```

## Models

### `GET /api/models`

MLflow/GridFS registry listing. `metrics` carries the honest-metrics block
(persistence/climatology baselines, skill score, Brier, prevalence, interval
coverage): every key is nullable and `null` when the registry document does
not carry it — never invented. `split`, `interval` and `commit` are passed
through from the registry document (also nullable).

```json
{
  "count": 2,
  "models": [
    { "name": "temp_prediction_1h_GradientBoostedTrees", "version": "20260923_020000",
      "target": "temperature", "horizon_hours": 1, "created_at": "2026-09-23T02:00:00Z",
      "metrics": { "rmse": 1.44, "mae": 1.12, "r2": 0.91,
                   "persistence_rmse": 3.05, "climatology_rmse": 3.41, "skill_score": 0.53,
                   "brier": null, "persistence_brier": null, "prevalence": null,
                   "coverage": 0.79 },
      "stage": "production",
      "split": { "kind": "temporal", "train_end": "2026-09-16T00:00:00Z",
                 "val_end": "2026-09-19T00:00:00Z", "test_start": "2026-09-19T00:00:00Z" },
      "interval": { "level": 0.8, "lower_offset": -1.9, "upper_offset": 2.1 },
      "commit": "abc1234" }
  ]
}
```

## Shared types

```ts
type CurrentWeather = {
  city: string;
  latitude: number | null;
  longitude: number | null;
  temperature: number | null;        // °C
  apparent_temperature: number | null; // °C
  humidity: number | null;           // %
  pressure: number | null;           // hPa
  wind_speed: number | null;         // km/h
  wind_direction: number | null;     // degrees, 0 = N
  precipitation: number | null;      // mm
  cloud_cover: number | null;        // %
  weather_code: number | null;       // WMO code
  observed_at: string;               // ISO-8601 UTC
};

type WeatherPoint = CurrentWeather;

type Prediction = {
  city: string;
  source_timestamp: string;
  prediction_timestamp: string;
  horizon_hours: number;
  predicted_temperature: number | null;
  predicted_rain: number | null;
  observed_temperature: number | null;
  temp_lower: number | null;         // predicted_temperature + lower_offset
  temp_upper: number | null;         // predicted_temperature + upper_offset
  interval_level: number | null;     // nominal level, 0.8 when present
  temp_model_name: string | null;
  temp_model_version: string | null;
  rain_model_name: string | null;
  rain_model_version: string | null;
};

type SeriesPoint = {
  timestamp: string;
  observed: number | null;
  model: number | null;
  aemet: number | null;
  residual_model: number | null;
  residual_aemet: number | null;
};

type Metrics = { mae: number | null; rmse: number | null; bias: number | null; n: number };
```

## Conventions

- Errors: `{ "detail": "human readable" }` with a 4xx/5xx status.
- CORS: allowed origins come from the `CORS_ORIGINS` env var (comma separated),
  default `http://localhost:5173`.
- No endpoint may return placeholder, synthetic or randomised values. If data
  is unavailable the field is `null` and the UI shows an explicit empty state.
- **Residual convention**: `residual_model` and `residual_aemet` are
  `prediction − observed`. Positive means the model ran warm, negative means it
  ran cold. `bias` follows the same sign.

## Station registry (v2.1)

The Mongo collection `cities` is the station registry and the source of truth
for ingestion: producers read it, the API exposes and manages it.

- `GET /api/cities` → array of `{ "name": string, "latitude": number, "longitude": number }`
  (the old bare string array is superseded; seeded with the 14 default
  stations on startup when the collection is empty).
- `POST /api/cities` body `{ "name", "latitude", "longitude" }` → 201 with the
  stored city. 409 when the name already exists (case-insensitive), 400 for
  invalid coordinates.
- `DELETE /api/cities/{name}` → 204. Removes the station from the registry
  only; historical documents in `raw_weather`/`weather_data` are kept.
- `GET /api/geo/search?q={query}` → `{ "results": [ { "name", "latitude",
  "longitude", "country", "admin1" } ] }` proxied from the Open-Meteo
  geocoding API (no key, `language=es`, max 8). `admin1`/`country` may be
  null. Upstream failure → 502 `{ "detail" }`; empty query → `{ "results": [] }`.
