# Handoff — operational notes for the next session

Read this **before** `docs/roadmap.md`. The roadmap holds the plan and the status
checklist; this file holds the environment, the hard rules and the traps that are not
written down anywhere else. Do not duplicate the roadmap here. Batch 4 frozen interfaces
live in `docs/batch4-contract.md`.

## Where things run

| Piece | Where | Notes |
| --- | --- | --- |
| Repo | `jomanor/weather-prediction-mlops`, branch `main` | Public repo → standard GitHub Actions minutes are unlimited. |
| API | Render service `meteoml-api`, deployed from `main` | `render.yaml`: docker, `backend/Dockerfile`, free plan, `autoDeploy: true`, health `GET /api/health`. Dashboard branch was the trap that caused stale deploys; it is now `main`. A deploy hook exists in the Render dashboard to force a deploy; the key is a secret, so it must never be committed to this public repo. |
| Frontend | Netlify site `meteoml` (`https://meteoml.netlify.app`) | Linked to `main` with base dir `frontend`, `npm ci && npm run build`, publish `dist`. `netlify.toml` is the source of truth (base, NODE_VERSION 22, headers, SPA redirects, `/api/*` edge proxy to Render). Auto publishing is currently **off** (`ignore = "exit 0"`); see the 2026-10-31 entry. |
| DB | Atlas M0 `weather_db` (512 MB) | Re-measured 2026-09-25: `collStats.avgObjSize` `raw_weather` 1362 B, `weather_features` 3856 B, `weather_predictions` **437 B** (Batch-3 interval columns; the old 405 B is stale), live city count **15** (includes Pasadena). Projected steady state **~380.5 MiB** against a 384 MiB (75 %) budget → **~3.5 MiB hard headroom**, after charging the L6 refresh endpoint's 48 accepted `raw_weather` writes/day against the same 180-day TTL. Prediction rows are keyed `(city, source_timestamp, horizon)` and written as upsert, so growth follows the ~6 h feature-rebuild cadence, not the hourly inference schedule; the storage-budget test's tripwire is **≥10 rebuilds/day** (8/day is modelled as cadence-jitter headroom; it was ≥14 before the refresh rows). `weather_features` is the largest collection and is a single overwritten snapshot, not an accumulator. The other breach axis is extending the raw TTL — re-measure before changing it. No TTL change in Batch 4: `weather_predictions` stays 90 d. |
| Cron | GitHub Actions | `ingest.yml` declared hourly `:50` (see the live caveat below); `ml-pipeline.yml` inference hourly `:05`, features `20 */6`, train nightly `35 2`; `backfill.yml` manual. |

Deleted branch: `rebuild/v2` is gone (it caused a stale Render deploy). `main` is the
only branch that matters.

Removed 2026-10-30: the legacy Kafka pipeline (`kafka/`, its compose services and CI
jobs) and the `torchcast/` notes folder — `scripts/ingest_weather.py` is the live
`raw_weather` writer.

## Commands

```bash
# Repo root. Python is the local venv; do not use system python.
.venv/bin/python ...                    # pymongo, requests, black, ruff, pytest
cd backend && ../.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest tests -q     # pipeline tests (spark/scripts)
cd frontend && npm run test && npm run typecheck && npm run lint && npm run build

# Backend against production data for a local check (kill it afterwards):
cd backend && MONGO_URI="$(grep '^MONGO_URI=' ../.env.production | cut -d= -f2- | tr -d '"')" \
  MONGO_DB=weather_db ../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8123

# Workflows
gh workflow run ml-pipeline.yml --ref main -f job=all        # features + train + prune + inference
gh workflow run ml-pipeline.yml --ref main -f job=train      # ~9 min
gh run watch <id> --exit-status
gh api repos/jomanor/weather-prediction-mlops/actions/jobs/<job_id> -q '.steps[]|.name+" "+.conclusion'
```

`.env.production` is unquoted and contains `&`, so `source` fails — parse the line as
shown above. **Never print `MONGO_URI`,** not even truncated. Local Mongo container
`weather-mongodb` runs on `:27017` if you need an offline database.

## Hard rules (these were paid for)

1. **Never run inference against stale `weather_features`.** Any change to feature
   semantics requires `job=all` (or features → train → inference) first; the saved model
   carries the feature list it was trained with. This is a *convention*, not a gate: the
   hourly `job=inference` schedule is the intended scoring loop and must keep working. The
   signal that catches a violation is the `$GITHUB_STEP_SUMMARY` block from
   `scripts/pipeline_summary.py` (features max timestamp + row count, model
   `data_snapshot.features`/`commit`), not a blocking check.
2. **Every Mongo sort must be prefixed by `city`** (`{city:1, timestamp:-1}`), and
   aggregates pass `allowDiskUse=True`. The `tag: 292` failures were an unindexed
   blocking sort. Never add a global `{timestamp:-1}` index. Do not rely on
   `allowDiskUse` alone — Atlas M0 rejects external sorting.
3. **Do not rename collections**, do not split `weather_data`/`raw_weather`, and do not
   change the prediction key `(city, source_timestamp, horizon_hours)` — model lineage and
   the backfill depend on them. (`weather_data` is now vestigial: `batch_processing.py`
   reads `raw_weather` and it currently holds 0 rows. Leave the collection and its TTL in
   place; do not "fix" it by writing to it.)
4. Charts read **oldest → newest**; the API reverses descending queries before returning.
5. One writer per tree. Concurrent work goes in separate worktrees (`git worktree add`),
   never the same file. The frontend agent found `node_modules` broken after a worktree
   install; `cd frontend && npm ci` fixes it.

## Verifying a deploy actually happened

Do not trust the dashboard. Compare the served asset hashes with a local build:

```bash
cd frontend && npm run build && ls dist/assets/*.js | xargs -n1 basename
curl -s https://meteoml.netlify.app/ | grep -oE 'assets/[A-Za-z0-9_.-]+\.js' | sort -u
```

Render: `GET /api/health/ready` predates Batch 3, so a 200 does **not** prove the new code
shipped. Use a Batch-3 endpoint instead — `GET /api/weather/quality?days=7` returning 404
means the old image is still running; 200 means the deploy landed. A `workflow_dispatch`
`job=inference` run is the cheapest way to exercise the full write path after a deploy.

To check that live payloads still satisfy the frontend contract, drop a temporary
`frontend/src/api/live-contract.test.ts` that fetches each endpoint and `safeParse`s it
with the schemas from `src/api/schemas.ts`, run it with `./node_modules/.bin/vitest run
<file>`, then delete the file. That is how a whole class of "the app is broken" reports
was ruled out.

## State at handoff (2026-10-19, after Batch 5)

- Batches 1–4 are merged on `main` (Batch 4 = 48b7380, PR #14). Batch 5 is merged and **pushed** (2026-10-30); the model-lifecycle contract
  lives in `docs/model-lifecycle.md`. The Batch-4 detail below is kept as history.
- **Batch 4 shipped (all four workstreams):**
  - **M2 horizons as config** (`spark/config/spark_config.py:37`): `FEATURES_CONFIG["target_horizons"] = [1,3,6,12,24]` replaces `target_horizon`; `ML_CONFIG["long_horizon_from"] = 12` uses a reduced grid (first value of each param) for `h >= 12`. One-pass `create_target_variable(df, horizons)`; `ml_training.main` caches the frame once and trains per horizon (per-horizon dropna/split/`data_snapshot`); `inference.main` loads once, loops horizons, skips a missing model with a warning, and unions all horizons into ONE upsert. `_id = {city}_{epoch(source_timestamp)}_{h}h`.
  - **M2 freshness fix**: `save_features_to_mongodb`'s `dropna` subset is the observation inputs only, so the newest observation row survives (it previously forced every healthy rebuild ~6 h stale). **Critical review fix:** `prepare_features_for_ml` now excludes **all** `target_*` columns (it previously kept the other horizons' labels, leaking future values and making inference write nothing).
  - **M4 diagnostics** (additive `model_registry.diagnostics`): `by_city`, `by_hour_of_day` (24, local hour), `by_rain_bucket` (dry <0.1 / light 0.1–2 / moderate 2–8 / heavy ≥8 mm), `drift_psi` (10 equal-width train bins, eps-guarded, null when <10 rows; a train→test proxy). Temperature slices fill rmse/mae/bias, rain slices brier, empty slices `n:0` + nulls. Exposed via `/api/models` (`ModelInfo.diagnostics`) and rendered as a ModelsPage "Deriva" badge (`<0.1` ok / `≤0.25` warn / `>0.25` drift, neutral when absent); `scripts/pipeline_summary.py` reports the top-3 per-feature PSI per model (informational) and emits a non-blocking `::warning` only on actionable degradation (`skill_score <= 0.0`), title `Model skill degradation`.
  - **Prediction API**: `GET /api/predictions/latest?horizon=` (omitted → one row per `(city, horizon_hours)`; provided → one row per city) and `GET /api/predictions/{city}?horizon=`; response shapes unchanged. Frontend horizon selector (1/3/6/12/24, default 1) on the station page + `useLatestPredictions(horizon)` on overview.
  - **D2 analytics**: six cached read-only endpoints under `/api/analytics` (daily, climatology, wind-rose, diurnal, correlation, error-by-hour), all on-demand over `weather_features`, TTL 300 s + ETag/304, no new collection and no Spark/CI stage. New frontend "Analítica" page (feature `analytics`, order 50). `error-by-hour` joins the observed temperature at the **target hour** `source_timestamp + horizon_hours`, not `prediction_timestamp`.
  - **U6**: chart brushing + synced x-domain via `useChartSync` (domain now reset on unmount); hand-rolled command palette (Cmd/Ctrl-K, focus trap, Escape); `g`+key section shortcuts; `/` focuses search; mobile station bottom sheet + swipe.
  - **U7**: fresh-data pulse keyed on data age (`useFreshness`, 120 min window matching hourly inference), shimmer skeletons, 120–240 ms transitions all gated by `prefers-reduced-motion`.
  - **Quality-meter age thresholds recalibrated** (`backend/app/repositories/weather_repo.py:182-183`): `QUALITY_OK_AGE_HOURS = 12.0` (was 2.0), `QUALITY_WARN_AGE_HOURS = 18.0` (was 6.0); completeness `0.95/0.80` deliberately unchanged. Rationale: the ~6 h rebuild cycle (plus GH jitter up to ~1.7 h) binds, so a healthy pipeline peaks ~12–14 h. The meter will still report `bad` on completeness once the one-shot backfill rolls off — the correct signal that the ingest cadence (L1) is unfixed.
  - **Cache hardening**: `cached()` coalesces concurrent cold calls per key via a shielded `asyncio.Task` (no key poisoning; a cancelling waiter cannot cancel the shared load); analytics computation bounded by `asyncio.Semaphore(2)`.
- **Gates**: `code-reviewer` found one blocker (the `target_*` leak above), fixed and re-reviewed CLOSED; `database-engineer` confirmed all new queries are index-backed with **no migration/index change**; `sre-engineer` confirmed thresholds and produced the storage projection; `qa-engineer` is **pending** (test-count gate below is from before its pass).
- **Test counts after remediation**: backend 191 passed; root 113 passed / 7 skipped; frontend 158 passed.
- **Live-data findings (Batch 4):**
  - Storage re-measured 2026-09-25: `weather_predictions` avgObjSize **437 B** (not 405) and **15** live cities (not 14) — see the DB row above for the projection and tripwire.
  - Predicted-row growth is bounded by the ~6 h feature-rebuild rate (upsert keyed on `source_timestamp`), not the hourly inference schedule.
  - ~~Pasadena is in Atlas but absent from `DEFAULT_CITIES`/`CANONICAL_CITIES`~~ **fixed**: `DEFAULT_CITIES` includes Pasadena and `analytics_repo.CANONICAL_CITIES` derives from it.
  - ~~`weather_features` is overwritten each rebuild, dropping its `{city:1,timestamp:-1}` index~~ **fixed in Batch 5 (M6)**: the write is append+upsert and `batch_processing.ensure_features_index` recreates the index after every write (warning-only on failure).
  - `weather_data` still holds 0 rows — expected; the R2 TTL entry for it is vestigial. Leave it in place.
- **Trap fixed in Batch 3, do not regress it:** `inference.py` used to trust the MLflow registry for artifact metadata; when MLflow served the artifact the row lost its `interval` and reported fake `MLflow-latest` lineage. `_latest_model_entry` is now resolved first and always supplies metadata; MLflow only ever serves the *artifact*.

## Resolved (2026-09-30) — `fix/spark-atlas-transient-retries` (Spark job resilience)

Merged to `main` as 8272b15. The evidence below is kept because the failure modes are still live. The agreed
follow-up shipped in Batch 5: `batch_processing.py` imports the shared `run_with_mongo_retries` and keeps no
local copy.

**Problem — `Run Spark jobs` is not "always failing", it fails ~5 % of runs but each one is a long red run.**
Workflow `ml-pipeline` on `main`, last 60 runs: 52 success / 3 failure / 5 cancelled. All 3 failures are the
`Run Spark jobs` job, all Atlas-side transients:

- run 60 (2026-09-29, inference): `com.mongodb.MongoSocketReadException: Prematurely reached end of stream`
  inside `write_predictions`' `output.count()` (py4j `o2362.count`).
- run 39 (2026-09-27, inference): same exception at `.save()` (py4j `o2380.save`).
- run 41 (2026-09-27, feature engineering, 31 s in): `MongoNodeIsRecoveringException` error 91
  (`ShutdownInProgress`, "quiesce mode") on shard `ac-ikelm1f-shard-00-02` plus a
  `MongoConnectionPoolClearedException` cascade.

**Why a single dropped socket ends the run, and why the window is hours wide:** (1) Spark runs in **local
mode** (`spark/config/spark_config.py`), so a stage aborts on the first task failure ("failed 1 times") —
there is no task-level retry — and `main` has no driver-side retry on the Mongo-touching actions;
(2) `inference.py` never caches `df_features`, so every action re-reads all of `weather_features` from Atlas
(~40 sequential Mongo stages). Identical code has measured **8–10 s per stage** (run 61 → 3 min job) and
**1,019–1,116 s per stage** (run 57 → 117 min; run 55 → 119 min) against the 120-min scheduled cap — the
Atlas M0 congestion tail *is* the failure window and two runs finished within 3 min of being killed.

**Branch contents (frozen interfaces):** new shared module `spark/spark-jobs/mongo_retry.py` —
`is_transient_mongo_error(exc)`, `retry_delay_seconds(attempt, base=5.0, factor=2.0, cap=60.0)`,
`run_with_mongo_retries(fn, *, label, attempts=4, sleep=time.sleep)`; it classifies pymongo errors **and**
`Py4JJavaError`-wrapped Java Mongo errors by class name / error code / message fragment (walks
`__cause__`/`__context__`, cycle-guarded) and returns False for anything unknown — fail fast, never mask a
real bug. `inference.py` caches `df_features` (one Atlas read per run, `unpersist()` in `finally`) and
retries the feature load, each per-horizon model load and the write (safe: `operationType=replace` +
`upsertDocument=true` is an idempotent `_id` upsert). `ml_training.py` retries the training feature read and
makes the GridFS upload / registry insert idempotent under retry (stable pre-generated `ObjectId`,
best-effort `fs.delete` on a failed attempt, duplicate key on retry = the file was already uploaded,
`replace_one(..., upsert=True)`). Worst case added wall time per retried operation is 35 s (4 attempts at
5/10/20 s), trivial against the 120-min cap.

**Deliberately NOT touched:** `spark/spark-jobs/batch_processing.py` — the uncommitted Batch-4 tree already
carries an equivalent helper inside it under the same public names. On merge, delete that local copy and
`from mongo_retry import run_with_mongo_retries` so feature engineering shares one implementation (the run-41
failure mode lives in that phase).

**Gates — both passed; every finding fixed (2 rounds each).** `code-reviewer` found one real must-fix: with a
deterministic GridFS `_id`, a lost final ack turns a transient fault into a permanent `DuplicateKeyError`
(11000, correctly classified non-transient) *and* orphans the file; the upload now treats a duplicate key with
`fs.exists(file_id)` as an already-completed upload. `qa-engineer` ran a mutation study: every retry boundary
in both jobs, every classification entry and the rewind-before-re-upload contract now fail when the
corresponding code is mutated. Its findings added a total `is_transient_mongo_error` (an unguarded `str(node)`
performed a JVM round-trip on a real `Py4JJavaError` and could mask the original error mid-classification),
an `attempts < 1` `ValueError`, a self-contained cache test and a real-handle upload test.

**Evidence:** `tests/unit` **163 passed / 1 skipped** (+48 over `main`'s 115 + 1 skip); `ruff check backend
scripts spark kafka` all checks passed; `black --check backend scripts spark kafka` 70 files unchanged (that
is the CI lint scope). Verify the branch with `git -C ../weather-prediction-mlops-atlas-retry show --stat HEAD`.

**Known non-blockers, deliberately not fixed:** `ml_training.main` never unpersists the feature frame it
caches (pre-existing, `spark.stop()` releases it); the 60 s backoff cap is unreachable at `attempts=4` (pinned
by a pure unit test only); `tests/` is outside CI's ruff/black scope; CI installs `pytest`/`pymongo` unpinned;
6 of the 8 integration tests need the CI `mongo:6` service and stay unmeasured locally.

**Status:** merged as 8272b15. No security gate was required (no auth, secrets or input surface changed).
The agreed follow-up (shared retry helper in `batch_processing.py`) is done; feature engineering, inference
and training now share one implementation.

## Batch 5 shipped (M5, M6, L6, R6)

- **R6 API surface**: every route is mounted once from an aggregate router under canonical `/api/v1`
  and a deprecated `/api` alias (Render's health path and the Netlify proxy keep working; the frontend is
  unchanged). Frozen error envelope `{"error":{"code","message","request_id"}}` for 404/422/500 and every
  explicit `APIError`; `/health/ready` keeps its documented 503 body. One JSON access line per request
  (`ts`, `level`, `request_id`, `method`, `path` without query string, `status_code`, `duration_ms`,
  `cache_hit`); `X-Request-Id` is echoed when it matches `[A-Za-z0-9._-]{1,64}` and regenerated otherwise.
  The TTL cache is now LRU-bounded and invalid cities are rejected against the registry.
- **M5 lifecycle** (`docs/model-lifecycle.md`): training always writes `staging`; `scripts/promote_model.py`
  promotes one document to `production`, supersedes the incumbent in the same slot (peers first, so a crash
  leaves zero production rather than two), supports `--dry-run` and `--rollback`, refuses a document whose
  GridFS artifact is missing, and is idempotent (a prefix re-run whose candidate is not newer than the
  incumbent exits 0 with "already production"). Inference resolves through
  `model_selection.select_entry`: production when present, otherwise the newest staging with a warning (this
  is why the hourly job keeps working until the first promote). Shadow A/B scores a strictly newer staging
  challenger on the same cached feature frame and upserts one `model_shadow` doc per (production, shadow)
  pair with disagreement stats — no extra prediction rows, never on the critical path, failures are logged
  and skipped. `scripts/prune_models.py` never deletes a `production` or `superseded` artifact, and removes
  shadow docs whose pair it pruned. `_id` epochs are pinned to UTC (`spark.sql.session.timeZone=UTC` +
  `utc_epoch_seconds`).
- **M6 canonical store**: GridFS artifact + `model_registry` metadata are the system of record; inference
  loads GridFS first and MLflow only for a legacy doc without a file id, never for metadata. MLflow logging
  is opt-in on `MLFLOW_TRACKING_URI`, imports lazily and cannot fail a run; `ml-pipeline.yml` no longer sets
  the CI file store or installs mlflow. The feature index is recreated after the write.
- **L6 refresh**: `POST /api/refresh` (and `/api/v1/refresh`) fetches one registered city from Open-Meteo,
  upserts the identical `raw_weather` document and returns
  `{city, source_timestamp, fetched_at, features_refreshed:false, predictions_refreshed:false}`. It never
  runs Spark. **Opt-in**: `REFRESH_ENABLED` defaults to `false` because this is an unauthenticated write
  surface. When enabled, a process-local limiter applies 900 s per city, 6/min, 48/day — rejections do not
  consume budget, attempts do; exceeded → 429 + `Retry-After`, disabled → 503, unknown city → 404, upstream
  → 502. Caps are per process lifetime and reset when the free-tier instance sleeps: they are a storage
  guard, not a distributed quota.
- **Follow-ups closed**: shared `mongo_retry` helper, LRU cache bound + city validation, feature-index
  repair, Pasadena in the derived city lists, tz-independent prediction `_id`, MLflow no longer a dependency
  of CI, mobile sheet/map touch fixes (tap tolerance, focus/Escape, swipe axis lock, safe-area layout).
- **Gates (measured on the merged tree)**: backend `235 passed`; root `315 passed, 7 skipped` (the 7 need a
  live `mongo:6`; run with `PYSPARK_PYTHON=$PWD/.venv/bin/python`); frontend `35 files / 187 tests passed`,
  typecheck clean, lint 0 errors, build ok (chunk-size warning only); `ruff` clean; `black` 81 files
  unchanged.
- **Review/QA outcome (fixed before merge)**: `reviewer` + `tester` independently found two blockers —
  `prune_models` deleted the promoted (oldest) artifact, and a prefix `promote_model.py` re-run promoted an
  older leftover staging doc. Both fixed with regression tests; also fixed: GridFS existence check on
  promote, demote-before-promote ordering, horizon-less slot matching, newer-only shadow challenger, storage
  budget including the 48 refresh docs/day (projection **380.5 MiB**, **3.5 MiB headroom**, first breaching
  feature-rebuild rate now **10/day**), and the docs' CLI examples. QA mutations caught 3 of 4 targeted
  faults; the surviving one (removing the limiter's `asyncio.Lock`) is unobservable because `acquire` has no
  `await` between check and increment — kept as defence-in-depth, docstring corrected.

**Decisions/assumptions to keep:** `/api` stays the served alias (removing it is a coordinated frontend +
Render change); the refresh endpoint is opt-in because it is unauthenticated, and its caps reset on instance
sleep; promotion stays a human CLI action, not an API; shadow A/B measures production↔challenger
disagreement, not accuracy, and only runs after a real promote.

**Deploy notes:** a fresh deploy serves both `/api/health` and `/api/v1/health` (either proving the image
shipped, unlike `/api/health/ready`); the hourly `job=inference` will log the staging-fallback warning until
the first promote; `POST /api/refresh` is available immediately but writes no features/predictions.


## 2026-10-30 — data-path incident and Batch 5.1 hotfix

Promoting the first models and exercising `job=inference` exposed a chain of real defects. All are fixed
and verified against production; do not regress them.

- **Features were duplicated 2x by legacy `_id`s.** `weather_features` held 43,823 ObjectId-keyed rows and
  43,851 string-keyed rows for the same `(city, timestamp)` — every legacy row had a string twin (verified
  1:1) because the deterministic `{city}_{epoch}` key was introduced after the backfill. Mixed `_id` types
  also made the Spark Mongo connector's `_id`-range partitioning read a **partial, run-varying** subset
  (73 % of rows, 11–15 cities), so inference/training silently saw different data each run. Fixed by
  deleting the 43,823 twin-verified ObjectId rows (recoverable: features rebuild from `raw_weather`) and
  adding `latest_per_city()` with `dropDuplicates(["city","timestamp"])` in `inference.py`.
  `weather_features` is now 43,880 rows, one per `(city, timestamp)`, all string `_id`. DB storage dropped
  to **173 MiB** (`dataSize` 241.8 MiB); the 380.5 MiB budget model is still the conservative bound.
- **GitHub throttles both hourly crons to ~4–7 runs/day** (evidence: `ingest.yml` runs at 22:00, 02:00,
  09:00, 16:15). Raw observations therefore arrive 6–7 h apart, so the 6 h rolling std features were null;
  `VectorAssembler(handleInvalid="skip")` then dropped every city and inference wrote only a stale Pasadena
  row (2026-09-25). Fixed by `coalesce(*_std_6h, 0.0)` (0 = "no observed variation in the window", not
  imputation), **a retrain is mandatory after this** (hard rule 1), plus an explicit `ingest_weather.py`
  step at the start of `job=inference`/`job=all`, and a `Scored N/M cities` line + non-blocking
  `::warning title=Prediction coverage` when <90 % of features cities are scored or the oldest source is
  >12 h behind. `spark/requirements.txt` now pins `pandas` (lazily-imported mlflow used to provide it).
- **Pasadena was missing from the live `cities` registry** (seed only runs when the collection is empty),
  so ingest fetched 14/15 stations and Pasadena's data froze at 2026-09-25. Re-added via
  `POST /api/v1/cities` (34.1478, -118.1445); ingest now stores 15/15. Its 12 h/24 h stds stay null until it
  has two observations a window apart, so it is the one city without a fresh prediction until then — the
  coverage ratio (14/15 = 93 %) deliberately does not warn for that single gap.
- **First real promotion.** `job=all` retrained on the new feature definition, then all 10 slots were
  promoted: 10 `production`, 8 `superseded`, 17 `staging`; inference loads them from GridFS and writes
  `temp_model_version=20261030_191959` etc. The 8 shadow docs written before the promote (old production vs
  new staging) are the first real A/B data. `temp_prediction_24h` (skill -0.117) and
  `rain_prediction_1h` (-0.044) are worse than persistence but are the best available; they are promoted so
  production matches the feature semantics.
- **Map bug fixed:** the "edificios 3D" layer never appeared because Carto renamed its vector source and
  building layer (`carto`/`building`/`render_height` → `protomaps`/`buildings`/`height`). `StationMap.tsx`
  resolves both pairs, inserts the extrusion under the labels, and only hides the flat footprints when the
  extrusion exists (`f03e1e9`).
- **After the hotfix:** predictions are fresh for 14/15 cities at the newest feature row (20:00), 70 rows,
  production model versions; `GET /api/v1/weather/quality` correctly reports `bad` completeness (~0.23)
  because the ingest cron still delivers only ~4–7 observations/day — that is the L1 signal, not a code bug.


## 2026-10-30 — P0: three surfaces, design system, city-write gate

The app was consolidated from five top-level surfaces to three; the proposal and
the cut list live in `docs/app-rethink.md`.

- **IA** — Resumen `/` (map + compact station list + one freshness line; only
  `/cities`, `/weather/current`, `/map/stations` on mount), Estación
  `/stations?city=X` (hero, one canonical chart with series toggles, one
  prediction card), Sistema `/sistema?tab=calidad|modelos|benchmark|analitica`
  (the former quality/models/benchmark/analytics pages as tabs). `/benchmark`,
  `/models`, `/analytics` redirect with `Navigate replace`; `/stations` was kept
  to preserve deep links. Nav is 3 items on both the top bar and the mobile
  bottom bar; shortcuts are `g r`, `g e`, `g s`.
- **Removed** — `StationSheet` (+ `sheet.ts`), `CityManager`, `DerivedPanel`,
  the per-station model name/version lines, the home MAE stat/quality meter, the
  city write hooks, and the unused `Sparkline`. Tapping a pin or a row now
  navigates to the station surface.
- **Design system** — one token set in `styles/index.css`: type 10/11/12/13/14/16/18/22
  px (old `text-4xl/5xl` clamped), radius 2/3/4 px, a single `0 1px 2px` shadow
  (plus the sheet elevation), 120/180/240 ms motion behind
  `prefers-reduced-motion`, one chart-axis/tooltip convention, and an accessible
  `Tabs` primitive. Banned: gradients/glows, icon soup, mixed chart styles,
  oversized heroes, inconsistent number formats. Font family unchanged (UI-03 is
  a pending user decision).
- **Security** — `POST /cities` and `DELETE /cities/{name}` now require
  `Authorization: Bearer $CITY_WRITES_TOKEN`; unset → 503 `city_writes_disabled`,
  wrong/missing → 401 `unauthorized` + `WWW-Authenticate`. Reads are unaffected.
  The public `POST /refresh` remains opt-in/off by default.
- **Data** — the 35-day hourly backfill fixed the 6 h gaps: 12,090 rows inserted
  (Pasadena 2 → 841 consecutive hours), quality went from completeness 0.216 /
  max-gap 8 h / all-`bad` to **1.000 / 1 h / all-`ok`** for 15 cities, and
  predictions now cover 15/15 cities fresh. Every feature rebuild also fills the
  last 48 h hourly, so the short-range charts stay dense between backfills.
- **Gates** — backend 258; root 315 + 7 skipped; frontend 222, typecheck, lint,
  build. Playwright probe on the deployed site: 3 nav items, no unnamed buttons,
  zero gradients, no horizontal page scroll at 390/1440, and the Resumen mount
  fires exactly the three expected endpoints. One real defect found by the probe
  (Analítica panned the page 238 px at 390 px because a sticky column inside the
  diurnal heatmap leaked into the root scroll area) is fixed with
  `position: relative` on the table.


## 2026-10-30 — P0 follow-up corrections (verified locally)

- **The type scale was silently stripped.** `cn()` uses `tailwind-merge`, which classifies
  `text-label|body|title|value|lead|page|hero` as *colours* and deletes them next to `text-fg*`.
  Every control therefore inherited the 16 px root (the "font is huge on PC" report). Fixed by
  registering the scale as `font-size` in `lib/cn.ts`; a unit test pins it. Verified computed sizes
  now: h1 18, panel title 13/14, body 12/13, segmented controls 11.
- **The Recharts `Brush` never mounted** (no `.recharts-brush` in the DOM), so the only chart
  interaction was the series toggles. Replaced with a controlled zoom/pan implementation
  (`chart-zoom.ts`, `useChartZoom.ts`): toolbar `− / + / Restablecer zoom`, wheel zoom anchored at
  the cursor, drag-to-pan, double-click reset, `touch-action: pan-y`, bounds clamped to the data +
  page window, all writing the shared `useChartSync` domain. Verified in a real browser: zoom-in
  moves the left edge later, pan shifts it, reset restores.
- **The station page showed the OLDEST prediction run and plotted predictions at
  `prediction_timestamp`.** The API's `/{city}` list is oldest→newest, `LatestPrediction` took
  `[0]`, and the chart drew each past run at the time inference ran. Now one unfiltered
  `/predictions/{city}?limit=200` query derives `latestRun` (max `source_timestamp`, one row per
  horizon); the card picks the selected horizon from it; the chart plots the run at
  `source_timestamp + horizon_hours`. Verified: the 48 h chart now spans observations + the +24 h
  forecast, the model line is in the future, and the card reads the latest run.
- **Pasadena is hidden from Sistema** (Calidad, Benchmark, Analítica) per the user's choice: the AEMET-covered set drives those tabs, so the city picker excludes it, `?city=Pasadena` falls back to Madrid, the diurnal heatmap has no Pasadena row and the quality table/counters are computed over the 14 Spanish stations (with a footnote pointing to Resumen/Estación). Resumen and Estación still show it; the API is unchanged.
- **Netlify**: `meteoml.netlify.app` returns Netlify's "Site not found" (404) while
  Netlify status is operational and Render is 200. The frontend fixes above were verified against a
  local Vite dev server proxying the live API. The deployed URL must be re-confirmed before trusting
  a live check.


## Next session (2026-10-31) — items 2 and 3 done, item 1 user-side

Live frontend: **https://meteoml.netlify.app** (base `frontend`, `npm ci && npm run build`, publish
`dist`; auto-publish is paused — see the 2026-10-31 entry). The old `weather-mlops-1787759329` URL was a
site rename. State at handoff: three surfaces, type scale applied (controls 11 px, titles 13/14, h1 18),
chart zoom/pan/reset, latest-forecast card/chart, Pasadena hidden from Sistema, CI green.

### 1. Netlify "Made with Netlify" badge / Drawer (user-side, no code)
- Badge: Netlify dashboard → **Site configuration → General → Netlify badge → Hide** (per-site; the
  `netlify.toml` has no option for this and CSS hacks are not supported).
- If the floating pill only appears while logged in, it is the **Netlify Drawer**: open it and choose
  "Hide Netlify Drawer" (account-level; visitors never see it). A custom domain does not remove either.
- If the badge toggle is plan-gated on that account, upgrading or hiding the Drawer is the only route.

### 2. Night icons (moon instead of sun) — **done 2026-10-31**
- Shipped as `is_day` end-to-end plus night descriptors/glyphs; details in the
  *2026-10-31 — Batch 6a* section below. Not live yet (auto-publish paused).

### 3. Mobile map controls cover the map — **done 2026-10-31**
- Shipped as `MapLayersControl.tsx` (`Capas` button + bottom sheet under `sm`, desktop panel
  unchanged); details in the *2026-10-31 — Batch 6a* section below. Not live yet (auto-publish paused).
- "Second tab" ambiguity **resolved**: Estación — reviewed at 390 px, no defect, no change.

### Verification recipe used today
`cd frontend && VITE_API_PROXY=https://weather-prediction-mlops.onrender.com npm run dev -- --port 4173`
then Playwright with the cached Chromium (`~/.cache/ms-playwright/chromium-1247/chrome-linux64/chrome`,
`playwright-core` installed in `/tmp/pw`). Check computed font sizes, `scrollWidth`, network paths,
zoom/domain changes and the station forecast timestamps. Deployed hash check:
`curl -s https://meteoml.netlify.app/ | grep -oE 'assets/index-[^.]+.js'` vs `ls frontend/dist/assets/index-*.js`.

## 2026-10-31 — Batch 6a: night icons + mobile map layers, deploys paused

- **Contract `is_day`**: `CurrentWeather.is_day` (`int | None`, 1 day / 0 night / null unknown) in
  `backend/app/schemas/weather.py`, mapped from `payload.current.is_day` in `_from_raw_weather`
  (`backend/app/repositories/weather_repo.py`) with `_RAW_FIELD_PATHS` and `WEATHER_FIELDS` entries;
  legacy `weather_data` leaves it null. `StationProperties.is_day` in `backend/app/schemas/map.py`,
  passed in `backend/app/routers/map.py` — without it the map glyph path falls back to the viewer's
  clock. Documented in `docs/api-contract.md`. **No pipeline rebuild was needed** and none is needed
  for this: `scripts/ingest_weather.py` already fetched `is_day`, so `raw_weather` already stored it.
  Do not retrain for this.
- **Frontend**: `describeWeather(code, isDay)` / `weatherIconName(code, isDay)` in
  `frontend/src/lib/weather.ts`, night descriptors for WMO 0/1/2/3 — `Cielo despejado (noche)`,
  `Mayormente despejado (noche)`, `Parcialmente nuboso (noche)`, `Cubierto (noche)`, tones unchanged;
  every other code keeps its day glyph. `is_day` parsed in `frontend/src/api/schemas.ts` as nullable
  0/1 (absent key → null, any other value → null). Call sites: `CurrentConditions.tsx`,
  `StationTable.tsx`, `components/map/station-geojson.ts`. The map needed no new artwork:
  `components/map/weather-icons.ts` rasterises the same lucide components, so the `moon`/`moon-star`/
  `cloud-moon` keys it already loops over are the night glyphs.
- **Viewer-local-hour fallback is deliberate.** In `lib/weather.ts` it is reachable only when `is_day`
  is genuinely null; it is a spec-mandated fallback and dead path once every payload carries the flag.
  A reviewer flagged it once. Do not "fix" it by guessing.
- **Test trap fixed**: `station-geojson.test.ts` asserted on a null-`is_day` fixture, so it was green
  or red depending on the runner's wall clock (UTC CI fails 22:00–06:59 UTC). Every day/night
  assertion now passes the flag explicitly; the suite is TZ-independent and passes under `TZ=UTC` and
  `TZ=Pacific/Auckland`.
- **Mobile map**: new `frontend/src/components/map/MapLayersControl.tsx` holds the layer settings once.
  Desktop `>= sm` keeps the unchanged `w-44` panel (`absolute right-2 top-2`); phones `< sm` get a
  44 px `Capas` button opening a bottom sheet with identical content (four layers, basemap, both
  legends), closing on selection, backdrop tap or Escape, 44 px tap rows, 16 px select (iOS zoom),
  focus to the close button and back to the trigger, Tab trapped, sheet offset by
  `calc(var(--nav-h) + var(--safe-bottom))` and capped with `100svh`. `StationMap.tsx` renders both
  surfaces and lets CSS pick. The 3D-buildings/protomaps source resolution, pin tap tolerance and
  radar controls are untouched. Browser-verified at 320/360/390/414/639/640 px.
- **Deploys paused**: `netlify.toml` has `ignore = "exit 0"` under `[build]`, so git-triggered Netlify
  builds are skipped (the every-push auto-publish was burning the Netlify tokens). A dashboard
  **Deploy site** or an API/build-hook deploy still builds; deleting that one line re-enables
  auto-publish. **The night-icon and `Capas` UI is committed but NOT live until such a deploy is
  deliberately triggered.**
- **Config**: `render.yaml` CORS now lists `https://meteoml.netlify.app` (commit 36b5f9c).
- **Gates measured on the merged working tree** (passes say nothing about correctness): backend
  `261 passed`, `ruff`/`black` clean; frontend 47 files / 323 tests passed, typecheck clean, lint
  0 errors (one pre-existing `vite.config.ts` warning), build ok.
- **Review**: an independent reviewer found one blocker — `/api/map/stations` was missing `is_day`,
  which made the moon glyphs unreachable in production and rendered Pasadena from the viewer's clock;
  fixed and re-reviewed CLOSED after the clock-dependent test fix.

## Next: Batch 6

No scope frozen yet. The cheapest remaining items are in `docs/roadmap.md` under *Accepted follow-ups*:
surface `X-Data-Age-Seconds` in the frontend (`apiGet` does not read response headers), consume
`/series`/`/summary`/`/range`, k6 cold-burst, a read-API latency/availability SLO (the JSON access log now
has the fields an OTel histogram would need), a pageable production-drift signal (sustained
`skill_score <= 0.0` or temperature-feature drift → GitHub issue, never a page), and a dead-man's switch for
`ingest` + features. Browser verification still needs the desktop app session.

**L1 (15-min ingest) is deferred, not cancelled.** Measured projection with 15-min ingest is ~637 MiB at a 180-day raw window (~890 MiB if features densify too), which breaks the 512 MB M0 cap. Three ways forward when it is picked up: shorten the raw window to ~90 d (free, ~350 MiB, halves the training window), upgrade the Atlas tier (2 GB, paid), or keep hourly ingest. Do not start L1 without re-measuring storage first.

Gate before done: `code-reviewer` on the diff, `qa-engineer` when tests change, `database-engineer`/`sre-engineer` for migrations and SLOs. Never waive a blocker.

## User-side items (cannot be done from the repo)

- **Render deploy hook secret.** Add repo secret `RENDER_DEPLOY_HOOK` (Render
  dashboard → `meteoml-api` → Settings → Deploy Hook) so CI's guarded `deploy`
  job can trigger a deploy after the tests pass on `main`. Without it the job
  skips with a notice and never fails. Separately, set the dashboard
  **Auto-Deploy to "On Commit"** (not "After CI Checks Pass"): with CI-checks
  mode Render skips the deploy when it sees zero checks on the merge commit.
  `render.yaml` already declares `autoDeployTrigger: commit`; the dashboard
  value must match. Running both is safe — Render ignores a deploy request for
  an already-deployed commit.
- Sentry DSN on Render.
- Healthchecks.io checks for the ingest/inference/train crons (highest value: a dead-man's switch on `ingest.yml` + features; the in-repo `::warning` is ephemeral).
- Atlas project alerts (disk >80%, connections >80%, query targeting).
- Browser verification needs this session open in the desktop app, otherwise the browser tools report `disconnected`.
