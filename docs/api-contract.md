# API contract v2

Single source of truth for the `backend/` service and the `frontend/` app.
Both sides must match this document. Changing it means changing both sides
in the same PR.

Canonical base path: `/api/v1` (Batch 5). The same route objects are also served under `/api` as a
deprecated alias — both answer identically, but new clients should use `/api/v1`. Render's health check and
the current frontend still use `/api`; removing the alias is a coordinated change.
Content type: `application/json`
Timestamps: ISO-8601 UTC strings (`2026-09-23T14:00:00Z`).
Nullable fields are always present in the payload, `null` when unknown —
never omitted.
Response headers: cached endpoints send `ETag`, `Cache-Control` and `X-Data-Age-Seconds`; every response
sends `X-Request-Id` (also embedded in error bodies).

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

Every `Prediction` also carries `source_age_hours` (number|null) and `stale`
(boolean): the age of `source_timestamp` at request time, and whether it exceeds
`PREDICTION_STALE_AFTER_HOURS` (env, default 24). Stale rows stay visible and
flagged — never dropped. A missing `source_timestamp` yields `null`/`false`.

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

## Metrics (process-local, public)

### `GET /api/v1/metrics` (alias: `GET /api/metrics`)

Aggregates the JSON access log into the surface an availability/latency SLO is
read from: totals, per-route latency estimates and one verdict. JSON, not
Prometheus text, and no new dependency (no exporter, no scraper, no store): the
access line already carries these fields per request, this endpoint is the
aggregate. It is unauthenticated and public-safe by construction (see below),
and `Cache-Control: no-store` — no `ETag`, no `X-Data-Age-Seconds`, because a
replayed copy would report a stale verdict.

```json
{
  "started_at": "2026-10-31T11:22:24Z",
  "uptime_seconds": 14.865,
  "requests": {
    "total": 9,
    "by_status_class": { "2xx": 8, "4xx": 1 },
    "errors_5xx": 0,
    "cache_hits": 1,
    "cache_hit_rate": 0.1111
  },
  "routes": [
    {
      "method": "GET",
      "route": "/api/v1/weather/history/{city}",
      "count": 2,
      "error_count": 0,
      "cache_hit_rate": 0.0,
      "p50_ms": 100.0,
      "p95_ms": 250.0,
      "p99_ms": 250.0,
      "max_ms": 111.4
    }
  ],
  "slo": { "latency_p95_ms_target": 2000, "p95_ms": 250.0, "met": true }
}
```

- **These counters are per process and reset on every restart.** On the free
  Render tier the instance sleeps when idle, so they also restart from zero when
  it wakes. They describe the current instance's life only — **not** a 28-day
  window — and must never be reported as one. A durable window needs an exporter
  or a store; deliberately out of scope.
- `requests.total` equals the sum of `routes[].count`. `by_status_class` lists
  only the classes seen since start (`"2xx"`, `"4xx"`, …). `errors_5xx` and
  `routes[].error_count` count **5xx only**: a 4xx is a client error, not an
  availability failure. `cache_hit_rate` is `null` when its denominator is 0,
  otherwise rounded to 4 decimals.
- `routes` is keyed by `(method, route template)` — `GET` ×
  `/api/v1/weather/history/{city}` — and sorted by `count` descending (ties broken
  by route, then method). Neither column can carry client text: `route` is a
  template the app declares (never the served path, so no city name), and
  `method` is normalized to a fixed set — `GET`, `HEAD`, `POST`, `PUT`, `PATCH`,
  `DELETE`, `OPTIONS`. **Any other method token** (a lowercase `get`, `TRACE`, a
  crafted `M4711` that uvicorn happily accepts) is reported as `<other>` and is
  never echoed back; `GET`, `HEAD` and `POST` on one path stay separate rows.
  Query strings, request ids and client addresses are never recorded either, so
  the payload is safe to serve publicly. (The JSON access log keeps the raw
  `method` verbatim — it is a server-side log line, not a response body.)
- **The table holds at most 256 entries, whatever the client sends.** The cap
  covers the whole table — every `(method, route)` combination and the reserved
  labels included — and one slot is always kept free, so up to 255 real rows plus
  the single overflow row exist. Anything past the cap is folded into one
  `(method: "<other>", route: "<other>")` entry, so neither method cardinality
  nor path cardinality can grow the map and the body stays under ~52 KB
  (`256 × ~200 B`). A request the router never matched is `<unmatched>` — a 404
  raised before routing, and a CORS preflight (answered by the CORS middleware
  without reaching a route) — and it is subject to the cap like any other row. A
  *routed* failure keeps its template: a 405 on `/api/v1/health` is counted
  there, and a `history` 404 for a city with no rows stays
  `/api/v1/weather/history/{city}`. `/api/v1/...` and `/api/...` are separate
  entries because they are separate mounts.
- **Percentiles are bucket upper bounds (no interpolation).** Latency is counted
  into fixed buckets — 5/10/25/50/100/250/500/1000/2500/5000/10000 ms — so memory
  is O(routes × buckets) and cannot grow with traffic. A reported `p95_ms` of 250
  means "the 95th percentile is at most 250 ms": a 120 ms request is reported as
  250, so the estimate is never below the true value. A sample above the last
  bound reports that route's `max_ms`, which is the largest duration actually
  observed. A route only appears once it has a sample, so the percentiles in the
  payload are never `null`; `null` is what the maths returns for an empty set.
- `slo.met` is the process-wide `p95_ms <= 2000` (reported alongside it), and
  `true` while no request has been recorded — nothing has violated the target
  yet. The bucket convention keeps the verdict conservative: a p95 above 2 s is
  bucketed at 2500 ms or higher, so `met` flips to `false` instead of being
  rounded down.
- `/metrics` is counted like every other route. The request that reads the
  payload is counted once its own response is complete, so it shows up from the
  next call on — two consecutive calls never show their own count.

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
  is_day: number | null;             // 1 = day, 0 = night, null when unknown
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

- Errors: every 4xx/5xx body is the stable envelope
  `{ "error": { "code": "snake_case", "message": "human readable", "request_id": "…" } }`.
  The only exception is `GET /health/ready` (503 `{ "status", "database" }`), an
  operational probe. Validation errors never echo the submitted value.
- CORS: allowed origins come from the `CORS_ORIGINS` env var (comma separated),
  default `http://localhost:5173`.
- No endpoint may return placeholder, synthetic or randomised values. If data
  is unavailable the field is `null` and the UI shows an explicit empty state.
- **Residual convention**: `residual_model` and `residual_aemet` are
  `prediction − observed`. Positive means the model ran warm, negative means it
  ran cold. `bias` follows the same sign.

## Station registry (v2.2)

The Mongo collection `cities` is the station registry and the source of truth
for ingestion: producers read it, the API exposes and manages it.

Reads are public. **Writes are gated** by the `CITY_WRITES_TOKEN` env var: the
registry drives ingestion, so deleting a station silently stops its ingestion
and degrades the quality meter. The gate applies identically to `/api/cities`
and `/api/v1/cities`.

- `GET /api/cities` → array of `{ "name": string, "latitude": number, "longitude": number }`
  (the old bare string array is superseded; seeded with the 15 default
  stations — 14 Spanish plus Pasadena — on startup when the collection is empty).
  Unauthenticated, never gated.
- `POST /api/cities` body `{ "name", "latitude", "longitude" }` → 201 with the
  stored city. 409 when the name already exists (case-insensitive), 400 for
  invalid coordinates.
- `DELETE /api/cities/{name}` → 204. Removes the station from the registry
  only; historical documents in `raw_weather`/`weather_data` are kept.
- Auth on both writes: `Authorization: Bearer <CITY_WRITES_TOKEN>`.
  - Token unset (also when the env var is empty) → **503**
    `city_writes_disabled`, message `City writes are disabled`.
  - Header missing, wrong scheme, or wrong token → **401** `unauthorized` with
    `WWW-Authenticate: Bearer`. The token is never echoed in the body or
    written to a log; the access log line carries no headers.
- `GET /api/geo/search?q={query}` → `{ "results": [ { "name", "latitude",
  "longitude", "country", "admin1" } ] }` proxied from the Open-Meteo
  geocoding API (no key, `language=es`, max 8). `admin1`/`country` may be
  null. Upstream failure → 502 `{ "detail" }`; empty query → `{ "results": [] }`.
