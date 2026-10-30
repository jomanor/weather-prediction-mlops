# Meteo platform roadmap

Goal: a **free**, highly customizable, comprehensive weather + ML platform.
Environment, hard rules and deploy-verification steps live in `docs/handoff.md` —
read that first.
Stack stays as-is (FastAPI/Mongo Atlas M0 on Render free, Vite/React on Netlify,
PySpark batch jobs on GitHub Actions). Public repo => standard Actions minutes are
unlimited, so cron density is free; the binding constraints are Atlas M0 512 MB,
the Render 750 h/month free pool and its ~15 min idle spin-down.

Effort: S < 1 day, M 1-3 days, L > 3 days.

## 0. Verified bugs (P0 correctness)

| ID | Bug | Evidence | Fix |
|----|-----|----------|-----|
| B1 | Wind shown ~3.6x too low | `scripts/ingest_weather.py:89` and `kafka/kafka-producer/producer.py:87` request `wind_speed_unit=ms`; `backend/app/schemas/weather.py:18` documents km/h; `frontend/src/lib/derived.ts:43` divides by 3.6 assuming km/h input | Convert m/s -> km/h at the API boundary in `weather_repo` mapping; ingest/producer/history untouched, no data migration |
| B2 | Time features use UTC hour | `spark/spark-jobs/batch_processing.py:127` `F.hour("timestamp")` | Derive local hour (offset from payload, else longitude/15) for `hour_sin/cos` and lag features |
| B3 | Registry says `PipelineModel` for every model | `spark/spark-jobs/ml_training.py:482` `model.__class__.__name__` | Thread the algorithm name (`GradientBoostedTrees`/`RandomForest`/`LinearRegression`) into `save_model` |
| B4 | Pasadena missing from maps/benchmark coverage | ~~Absent from the ingest city list~~ **Not a bug:** `scripts/ingest_weather.py:76` reads the Mongo `cities` collection, so Pasadena is ingested. It is absent from the AEMET benchmark because AEMET has no US coverage | No code change; document that the benchmark is Spain-only |

Note: ingest and AEMET coverage of Pasadena is by design, not a defect.
| B5 | History charts run newest -> oldest | feature chart series order | Return/render series oldest -> newest |
| R1 | Inference appends instead of upserting | `spark/spark-jobs/inference.py` write path `mode("append")` while the docstring claims upsert | Stable `_id` = `{city}_{epoch(source_timestamp)}_{horizon}h` + `replaceDocument=true`; dedupe legacy rows with `scripts/dedupe_predictions.py` |

## 1. Foundation - modularity + contract

| ID | Item | Files | Effort |
|----|------|-------|--------|
| F1 | Feature-module registry: each feature declares route + nav + queries; `App.tsx`/`AppShell.tsx`/`nav.ts` become registry-driven; `React.lazy` per route | `frontend/src/app/{registry.ts,routes.tsx}`, `app/nav.ts`, `app/AppShell.tsx`, `features/*/index.ts` | M |
| F2 | Split `api/queries.ts` per feature; shared hooks `useUrlState`, `useStationSelection`, `useChartSync`, `useVisibilityRefetch` | `frontend/src/hooks/`, `api/queries.ts` | M |
| F3 | Contract generation: `openapi-typescript` -> `api/generated.ts` + CI drift check; Zod schemas typed against generated types | `frontend/src/api/`, `scripts/check_contract.py`, `ci.yml` | S-M |
| F4 | Backend: move `seed_default_cities` to `services/`, add `?horizon=` to the predictions route. Do **not** restructure the router/repo layout | `repositories/city_repo.py`, `routers/predictions.py` | S |

## 2. Freshness / "live" feel

| ID | Item | Effort |
|----|------|--------|
| L1 | Ingest hourly -> 15 min; features 6-hourly -> hourly (the 6 h staleness source); chain inference on ingest via `workflow_run` | M |
| L2 | In-process TTL cache + `ETag`/`Cache-Control`/304 on `/current`, `/history`, `/stats`, `/benchmark`; bucket the `now` anchor per hour | S |
| L3 | Slim cold start: `ensure_indexes`/seed after `yield` | S |
| L4 | `GET /api/weather/series?cities=&from=&to=` (bulk, projection), `GET /api/map/stations` GeoJSON, `GET /api/weather/summary` for sparklines | M |
| L5 | `GET /api/weather/range/{city}?from=&to=` (removes the 168 h / 500-point ceiling); `X-Data-Age-Seconds` header | S |
| L6 | Optional `POST /api/refresh` (rate-limited, server-side fetch, never a browser-triggered workflow dispatch). SSE only as polish; **WebSocket rejected** | M |
| L7 | Prefetch + focus refetch + skeletons; keep-warm pinger only if the Render 750 h pool is otherwise unused | S |

## 3. Frontend - map, colour, interactivity

| ID | Item | Effort |
|----|------|--------|
| U1 | Cross-filtering: map pin <-> table row <-> chart, one lifted `selectedCity` | M |
| U2 | URL-synced state (`city`, `hours`, `var`, `range`, `tab`) | M |
| U3 | Radar/satellite playback with a time scrubber (RainViewer, keyless) | M |
| U4 | Data-ramp tokens (`--temp-*`, `--rain-*`, `--wind-*`); temperature-coloured markers, wind arrows | S |
| U5 | Clustered GeoJSON markers; basemap switcher (positron/dark/OpenTopoMap) | M |
| U6 | Chart brushing + synced x-domain; command palette; keyboard shortcuts; mobile swipe/bottom-sheet | M each |
| U7 | Weather-code icons; fresh-data pulse; shimmer skeletons; 120-240 ms transitions honouring `prefers-reduced-motion` | S |

Prerequisite: no `frontend.config.json` yet - pin design tokens before implementing.

## 4. ML platform

| ID | Item | Effort |
|----|------|--------|
| M1 | Persist metrics + feature importances + tuned params + commit + data snapshot in `model_registry` | S |
| M2 | Horizons `[1,3,6,12,24]` as config: one target pass, loop training/inference; reduced grids for long horizons | M |
| M3 | Prediction intervals (ensemble spread / residual quantiles) + measured coverage in the UI | M |
| M4 | Slice diagnostics (`by_city`, `by_hour_of_day`, `by_rain_bucket`) + drift PSI -> badge + workflow warning | M |
| M5 | Stages (`staging` -> human promote -> `production`), rollback script, shadow A/B | M-L |
| M6 | Declare GridFS + `model_registry` canonical; the CI MLflow file store is ephemeral | S |

## 5. Data science - trust + analytics

| ID | Item | Effort |
|----|------|--------|
| D1 | Temporal splits + persistence/climatology baselines; the current random split + `lag_1h` makes R^2 0.984 a persistence artifact. Rain: AUC-PR + Brier + prevalence | M |
| D2 | Analytics collection + endpoints: anomaly vs climatology, HDD/CDD, heatwave detection, wind rose, diurnal heatmap, city correlation, error-by-hour | M |
| D3 | Verification loop: join prediction vs observed at t+h -> `prediction_errors`; rolling MAE vs persistence; exclude ERA5-backfilled rows | S |
| D4 | Data-quality meter per city (completeness, max gap, null rate, last-hour age) | M |

## 6. Reliability / ops / CI

| ID | Item | Effort |
|----|------|--------|
| R2 | TTL: `raw_weather` 180 d, `weather_predictions` 90 d, `weather_data` 30 d -> steady state ~235 MB of 512 MB | S |
| R3 | Indexes: `weather_features` has **none** -> add `{city:1,timestamp:-1}`; add `{city:1,horizon_hours:1,prediction_timestamp:-1}`; every sort `city`-prefixed + `allowDiskUse` + `maxTimeMS` | S |
| R4 | CI guardrails: real `mongo:6` index-backed-sort test, inference output-schema contract test, storage-growth budget, `prune_models` `if: always()` | M |
| R5 | Observability: Sentry free, Healthchecks.io dead-man's-switch on the crons, Atlas alerts (disk/connections/query targeting), `GET /api/health/ready` | S |
| R6 | Structured JSON logs with `request_id`/`duration_ms`/`cache_hit`; stable error envelope; `/api/v1` mount (breaking - do before the frontend hardens) | M |

## 7. Explicitly cut

Paid tile providers; fake animated wind fields (no gridded wind data exists);
WebSocket live on a sleeping instance; Redis/ClickHouse/event-sourcing/lakehouse;
microservices; hosted MLflow; Airflow/Evidently/KServe/Feast; a second icon set;
gamified badges; per-city quantile models. Do not split `weather_data`/`raw_weather`,
rename collections, or change the prediction key `(city, source_timestamp)` -
that invalidates model lineage and forces a re-backfill.

## Sequencing

1. **Correctness:** B1-B5, R1, R3, R2.
2. **Foundation:** F1, F2 -> L2, L3, L4.
3. **Interactive UI:** U1, U2, U3 -> U4, U5 -> U6, U7.
4. **Trust:** D1, M1, D3 -> M3, D4.
5. **Richness:** M2, D2, M4 -> M5, L6, R4-R6.

## Status

- [x] PR #6 repopulate 90 d history; PR #7 blocking-sort fix; PR #8 training + inference fixes; PR #9 Modelos tab null metrics
- [x] Batch 1 (PR #10): B1 wind km/h at the API boundary, B2 local-hour time features, B3 real `model_type`, B4 clarified as by-design, B5 chronological chart series, R1 inference upsert (connector `operationType=replace` + `upsertDocument`, 15 legacy duplicates removed), R2 TTL (raw 180 d / predictions 90 d / weather_data 30 d), R3 missing `weather_features` + predictions indexes, R5 `GET /api/health/ready`
- [x] Batch 2 (PR #11 backend, PR #12 frontend): F1 feature-module registry + lazy routes, F2 per-feature queries + shared hooks (`useUrlState`, `useStationSelection`, `useChartSync`), L2 in-process TTL cache + ETag/304, L3 background index/seed so a cold start serves immediately, L4 `GET /api/weather/series` + `/api/weather/summary` + `/api/map/stations` (GeoJSON), L5 `GET /api/weather/range/{city}` (keyset), U1 map/table/chart cross-filtering, U2 URL-restored view state
- [x] Batch 3 (PR #13): U3 radar playback with time scrubber, U4 data-ramp tokens (`--temp-*`/`--rain-*`/`--wind-*`), U5 clustered GeoJSON stations + weather-code icons + wind arrows + basemap switcher, D4 data-quality meter (`GET /api/weather/quality`), M1 honest metrics (temporal split by timestamp value, persistence/climatology baselines, skill score, `data_snapshot`/`commit`), M3 prediction intervals (residual quantiles, live in `weather_predictions`), R4 CI guardrails (real `mongo:6` index-sort test, inference output-schema contract, storage budget, `prune_models if: always()`, `scripts/pipeline_summary.py`)
- [x] Batch 4 (branch `feat/batch4`, uncommitted at handoff; frozen interfaces in `docs/batch4-contract.md`): **M2** horizons `[1,3,6,12,24]` as config (`target_horizons`, one-pass `create_target_variable`, reduced grid for `h >= long_horizon_from=12`), per-horizon training/inference unioned into one upsert (`_id = {city}_{epoch(source_timestamp)}_{h}h`), freshness fix so `save_features_to_mongodb` no longer drops the newest observation, and `prepare_features_for_ml` now excludes **all** `target_*` columns (the leak that made inference write nothing); **M4** `model_registry.diagnostics` (`by_city`, `by_hour_of_day` 24 local hours, `by_rain_bucket` dry/light/moderate/heavy, `drift_psi` train→test proxy) exposed via `/api/models` and a ModelsPage "Deriva" badge, plus a non-blocking `::warning` in `scripts/pipeline_summary.py` only on `skill_score <= 0.0` (top-3 per-feature PSI reported informationally); **prediction API** `GET /api/predictions/latest?horizon=` (all horizons when omitted, per-city when provided) and `/{city}?horizon=`, with a frontend horizon selector (default 1); **D2** six cached read-only `/api/analytics/*` endpoints (daily, climatology, wind-rose, diurnal, correlation, error-by-hour) over `weather_features` with TTL 300 s + ETag/304, no new collection, and an "Analítica" page (order 50); **U6** chart brushing + `useChartSync` domain reset on unmount, hand-rolled command palette (Cmd/Ctrl-K, focus trap, Escape), `g`+key shortcuts, `/` search focus, mobile bottom sheet + swipe; **U7** fresh-data pulse keyed on data age (`useFreshness`, 120 min), shimmer skeletons, 120–240 ms transitions gated by `prefers-reduced-motion`; quality-meter age thresholds recalibrated to `12.0/18.0 h` (completeness `0.95/0.80` unchanged); cache single-flight per key (shielded `asyncio.Task`) + `asyncio.Semaphore(2)` bound on analytics; storage re-measured (predictions 437 B, 15 live cities) → ~368.7 MiB projected at 8 feature rebuilds/day vs the 384 MiB budget, first breach at ≥14 rebuilds/day. **L1 still deferred.**
- [x] Batch 5 (merged and pushed 2026-10-30): **M5** `staging`→human promote→`production`→`superseded` via `scripts/promote_model.py` (promote/rollback/dry-run, refuses a doc without `gridfs_file_id`, idempotent), `model_selection.select_entry` shared by inference and the script (production, else newest staging + warning so the hourly job survives before the first promote), bounded shadow A/B (one `model_shadow` doc per production/shadow pair, disagreement not accuracy, never on the critical path), tz-pinned prediction epoch; **M6** GridFS artifact + `model_registry` metadata declared canonical in `docs/model-lifecycle.md`, inference GridFS-first, MLflow opt-in/lazy and dropped from `ml-pipeline.yml`, feature `{city:1,timestamp:-1}` index recreated after the write; **L6** `POST /api/refresh` + `/api/v1/refresh` server-side Open-Meteo fetch (no workflow dispatch, no WebSocket), opt-in via `REFRESH_ENABLED` with 900 s/city, 6/min, 48/day per-process limits, `raw_weather` upsert only, honest `features_refreshed:false`/`predictions_refreshed:false`; **R6** `/api/v1` canonical + deprecated `/api` alias from one aggregate router, stable error envelope, one JSON access line (`request_id`/`duration_ms`/`cache_hit`, no query string), `X-Request-Id` validation, LRU-bounded cache. Gates after review fixes and the 2026-10-30 hotfix: backend 235, root 315+7 skipped, frontend 189, ruff/black/build green; `prune_models` now retains production+superseded artifacts and `promote_model.py` prefix re-runs are a clean no-op; storage projection 380.5 MiB of the 384 MiB budget (3.5 MiB headroom, refresh rows included, first breach at 10 feature rebuilds/day).
- [x] Batch 5.1 hotfix (2026-10-30, deployed): removed 43,823 twin-verified legacy ObjectId `weather_features` rows (mixed `_id` types made the Mongo connector read a partial, run-varying subset), added `latest_per_city` dedupe, zero-filled the 6 h rolling stds + retrained + promoted all 10 slots, ran ingest inside the inference job, added the prediction-coverage warning, restored `pandas`, re-added Pasadena to the live registry, and fixed the Carto 3D-buildings layer. Details in `docs/handoff.md`.

## Accepted follow-ups

Found in review, deliberately deferred (none blocking):

- ~~`cached()` is single-flight per key as of Batch 4, but cache keys come from an unbounded `city` string~~ **done in Batch 5**: the TTL cache is LRU-bounded and cities are validated against the registry.
- `weather_data` is now vestigial (0 rows; `batch_processing.py` reads `raw_weather`), so its 30-day TTL no longer truncates anything. Leave the collection and TTL in place.
- ~~`model_repo.list_models` sorts on `timestamp` alone~~ **fixed in Batch 3**: sort is now `[("model_name", 1), ("timestamp", -1)]`, and the R4 `mongo:6` test asserts the app-declared city-prefixed index.
- ~~The quality meter's age thresholds (≤2 h ok / ≤6 h warn) are calibrated to an hourly ingest the system does not currently achieve~~ **done in Batch 4**: recalibrated to `12.0/18.0 h` against the ~6 h rebuild cycle. Completeness `0.95/0.80` stays; the meter will still report `bad` on completeness once the one-shot backfill rolls off — that is the correct signal that the L1 cadence is unfixed.
- ~~`core/cache.py` `cached()` has no single-flight~~ **done in Batch 4**: concurrent cold calls for one key now share a shielded `asyncio.Task` (no key poisoning; a cancelling waiter cannot cancel the shared load).
- ~~`useChartSync`'s shared domain is set by data loads and never reset, so a chart can transiently inherit another page's window~~ **fixed in Batch 4**: the shared domain resets on unmount.
- ~~`CityManager` deletes a station without confirmation~~ **done** (4cd4caa).
- `useUrlState`'s casts are sound only because one concrete schema drives each param key; nothing prevents two surfaces from reusing a key.
- The frontend still does not consume `/series`, `/summary` or `/range` (the map reads `/api/map/stations`).
- ~~`weather_features` loses its `{city:1,timestamp:-1}` index because `ensure_indexes` only runs at backend startup~~ **done in Batch 5 (M6)**: the write is append+upsert and `batch_processing.ensure_features_index` recreates the index after every write (warning-only on failure).
- ~~Pasadena is present in Atlas (15 cities) but absent from `DEFAULT_CITIES`/`CANONICAL_CITIES`~~ **fixed**: `DEFAULT_CITIES` includes Pasadena and the derived lists follow it.
- ~~`weather_predictions._id` uses `unix_timestamp`, which is session-TZ-dependent~~ **done in Batch 5**: the session TZ is pinned to UTC and the epoch is computed by `utc_epoch_seconds`, covered by a non-UTC-TZ test.
- The frontend freshness pulse keys on payload timestamps because `apiGet` does not surface response headers (`X-Data-Age-Seconds` is unused).
- The refresh endpoint's caps are per process and reset when the free Render instance sleeps; enabling it is a deliberate operator choice. A distributed quota would need a Mongo-backed counter.
- **GitHub Actions throttles the declared hourly crons to ~4-7 runs/day**, so raw observations arrive 6-7 h apart and the quality meter reports `bad` completeness (~0.23). This is the L1 problem in practice, not an extra one: any cadence fix must be verified against the actual schedule, not the cron string.
- No k6 cold-burst test yet: 3 analytics views + concurrent `/api/health`, recording p95 and health-check failures.
- No read-API availability/latency SLO (none exists); needs an OTel histogram or structured slow-request logs first (target e.g. p95 < 2 s, 99 %/28 d).
- No production-drift SLI: PSI proved non-actionable (uniform ~0.51 from seasonal `pressure`), so alert on sustained `skill_score <= 0.0` (or `temperature`-feature drift) via a GitHub issue (never a page). The existing `::warning` is ephemeral/not pageable. Higher value: a pipeline dead-man's switch for `ingest.yml` + features (Healthchecks.io is still user-side).
- **L1 (15-min ingest) deferred**: projects ~637 MiB vs the 512 MB M0 cap. Requires a storage re-measure and a user decision before starting; do not start it without one.
