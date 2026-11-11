# Handoff — operational notes for the next session

**Start here:** `docs/audit-2026-10-31.md` is the queued work list (20 findings + 12 smaller
items, batches A–F, and the owner decisions). Batches A, B and C are done (C is local and uncommitted as
of 2026-11-01), and finding 7 (`POST /refresh`) is resolved by deletion (`decaba5`). Pick a batch from its
*Suggested order* table, fall back to `docs/roadmap.md` *Accepted follow-ups* when it is exhausted. **Next
is D** — which needs the owner's Analítica and Pasadena answers — **or E**, which needs the Spark answer.
Do not start a batch before answering the owner questions that batch depends on.

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
| DB | Atlas M0 `weather_db` (512 MB) | Re-measured 2026-09-25: `collStats.avgObjSize` `raw_weather` 1362 B, `weather_features` 3856 B, `weather_predictions` **437 B** (Batch-3 interval columns; the old 405 B is stale), live city count **15** (includes Pasadena). Projected steady state **~380.5 MiB** against a 384 MiB (75 %) budget → **~3.5 MiB hard headroom**, after charging the L6 refresh endpoint's 48 accepted `raw_weather` writes/day against the same 180-day TTL (that endpoint was deleted 2026-10-31, so the charge is spare headroom now, not a cost). Prediction rows are keyed `(city, source_timestamp, horizon)` and written as upsert, so growth follows the ~6 h feature-rebuild cadence, not the hourly inference schedule; the storage-budget test's tripwire is **≥10 rebuilds/day** (8/day is modelled as cadence-jitter headroom; it was ≥14 before the refresh rows). `weather_features` is the largest collection and is a single overwritten snapshot, not an accumulator. The other breach axis is extending the raw TTL — re-measure before changing it. No TTL change in Batch 4: `weather_predictions` stays 90 d. |
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

# Backend against production data for a local check (kill it afterwards). The
# single repo-root .env already points MONGO_URI at Atlas; backend/ reads it via
# the env_file=(".env", "../.env") tuple, so nothing has to be exported:
cd backend && ../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8123
# (port 8123 is often taken by a hand-run instance: pick another, the check is
# the same. curl /api/v1/metrics and read database.indexes to prove the code shipped.)

# Workflows
gh workflow run ml-pipeline.yml --ref main -f job=all        # features + train + prune + inference
gh workflow run ml-pipeline.yml --ref main -f job=train      # ~9 min
gh run watch <id> --exit-status
gh api repos/jomanor/weather-prediction-mlops/actions/jobs/<job_id> -q '.steps[]|.name+" "+.conclusion'
```

**One env file: `.env` at the repo root** (gitignored, mode 600). It replaced
`.env` + `.env.production` + `atlas-credentials.env` on 2026-10-31 — see the
*Batch B* section. It holds `MONGO_DB`, the Atlas `MONGO_URI`, the local mongo
`MONGO_USERNAME`/`MONGO_PASSWORD` (docker-compose only) and `AEMET_API_KEY`.
Never `source` it (the password contains `&`) and **never print `MONGO_URI`**,
not even truncated. Production config is not a file: `render.yaml` + the Render
dashboard for the API, the Netlify UI for the frontend; the tracked placeholder
for onboarding is `.env.example`. Local Mongo container `weather-mongodb` runs on
`:27017` if you need an offline database.

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
  - **U6**: chart brushing + synced x-domain via `useChartSync` (domain now reset on unmount); hand-rolled command palette (Cmd/Ctrl-K, focus trap, Escape); `g`+key section shortcuts; `/` focuses search; mobile station bottom sheet + swipe. **(Palette and `g` chords deleted 2026-11-01, audit S11; `/` and Cmd/Ctrl-K now focus `StationSearch`.)**
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
- **L6 refresh** (removed 2026-10-31 — see `docs/audit-2026-10-31.md` finding 7): `POST /api/refresh`
  (and `/api/v1/refresh`) fetched one registered city from Open-Meteo, upserted the identical `raw_weather`
  document and returned
  `{city, source_timestamp, fetched_at, features_refreshed:false, predictions_refreshed:false}`. It never
  ran Spark. **Opt-in**: `REFRESH_ENABLED` defaulted to `false` because this was an unauthenticated write
  surface. When enabled, a process-local limiter applied 900 s per city, 6/min, 48/day — rejections did not
  consume budget, attempts did; exceeded → 429 + `Retry-After`, disabled → 503, unknown city → 404, upstream
  → 502. Caps were per process lifetime and reset when the free-tier instance slept: they were a storage
  guard, not a distributed quota. **The owner chose deletion over gating (`decaba5`)**; nothing replaced it.
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
Render change); promotion stays a human CLI action, not an API; shadow A/B measures production↔challenger
disagreement, not accuracy, and only runs after a real promote. The refresh endpoint was **deleted**
(2026-10-31), so its opt-in/limiter caveat no longer applies to anything.

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
  bottom bar; shortcuts are `g r`, `g e`, `g s` (deleted 2026-11-01, audit S11 —
  `/` and Cmd/Ctrl-K focus the one `StationSearch`).
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
  The public `POST /refresh` that used to sit beside it was deleted on 2026-10-31 (`decaba5`).
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

## 2026-10-31 — Batch 6b: pipeline watchdog + server-side data age

Both halves are in the working tree and **not on `main`** — `scripts/watchdog.py`,
`.github/workflows/watchdog.yml` and `tests/unit/test_watchdog.py` are untracked, the frontend files are
modified-but-uncommitted. The cron does not exist until the workflow lands on `main`, and auto-publish is
paused (2026-10-31 entry), so neither half is live.

**Watchdog — `scripts/watchdog.py` + `.github/workflows/watchdog.yml`**

- Stdlib only (`urllib.request`, `json`, `argparse`, `dataclasses`), no `pip install`: it must run before
  any dependency install. Reads the public read API only (`/api/v1/weather/current`,
  `/api/v1/weather/quality?days=7`, `/api/v1/predictions/latest`, `/api/v1/models`) — no Mongo
  credentials, no `gh` CLI, and nothing writes to the database or the pipeline.
- One rolling issue, no fan-out: title **`[ops] Pipeline de datos desactualizado`**, label **`ops-alert`**
  (created itself on a 422 when the label is missing), matched by *title* so a hand-removed label cannot
  fork a second one. A `<!-- watchdog-verdict {...} -->` block in the body/comment makes it idempotent:
  unchanged verdict → no comment, recovery → one comment + close. At most one issue, ≤1 comment per run.
- Cron `25 */3 * * *` plus `workflow_dispatch`; `concurrency: watchdog`, `cancel-in-progress: false`;
  `permissions: contents: read` + `issues: write`; Python 3.12; `timeout-minutes: 15` (backstop above
  the code's own budget: 360 s evidence budget + one 120 s attempt, plus the GitHub calls); one run step.
  **No secret beyond the automatic `GITHUB_TOKEN` and no external account** — it replaces the user-side
  Healthchecks.io ping.
- Thresholds are *mirrored*, never invented: newest `observed_at` and newest `source_timestamp` 12 h ok /
  18 h warn-or-bad, worst-city completeness bad `< 0.80`, prediction coverage bad `< 0.90`, edges
  inclusive (14/15 = 0.933 stays ok). They are one block at the top of `scripts/watchdog.py`, copied from
  `backend/app/repositories/weather_repo.py` (12.0/18.0) and `scripts/pipeline_summary.py`
  (`PREDICTION_COVERAGE_WARN_RATIO`). **Retune the backend first and copy the numbers here; never retune
  the switch alone**, or it will disagree with the quality meter.
- **Model skill degradation is advisory only and must stay that way.** Production models at or below a
  `skill_score` of `0.0` are printed in the issue body but never create an issue or a comment on their
  own, because two promoted models already sit below zero (`temp_prediction_24h_LinearRegression -0.117`,
  `rain_prediction_1h_GradientBoostedTrees -0.044`, 2026-10-30 entry). Reading the live payload, the level
  is `stage` (or `status`) and the score is `metrics.skill_score` (falling back to
  `diagnostics.skill_score`); null/absent stays "unknown" and is never advisory.
- **Green vs red.** Exit `0` means "the watchdog worked": everything in threshold, the issue was
  created/commented/closed, **or the read API could not be read**. A red run (exit `1`) means the
  **watchdog itself is broken** — missing `GITHUB_TOKEN`/`--repo`, a GitHub REST error, or an internal
  exception including a payload that breaks the documented API shape. Read the *issue*, not the check.
- **An unreachable API is a verdict, not a red run.** A timeout or connection fault after the retries (and
  any 5xx/502, e.g. during a Render deploy) on a graded endpoint becomes the `api_down` verdict: it opens
  or updates the same rolling issue with the error text in the body, and the run still exits `0`. A
  failing `/api/v1/models` is narrower: it records `advisory_error`, prints it in the body and never
  aborts the run, so a sick advisory endpoint cannot mask a stale pipeline.
- **Timeouts are sized for the Render cold start, not to make the run red:** `HTTP_TIMEOUT_SECONDS =
  120.0` per read-API attempt (3 attempts, 5 s / 10 s backoff) inside `API_TOTAL_BUDGET_SECONDS = 360.0`
  for the whole evidence phase (≈8 min worst case, after which remaining calls fail instantly as
  `api_down`/`advisory_error`); GitHub calls keep their own `GITHUB_TIMEOUT_SECONDS = 30.0` because there
  is no cold start there. The old 30 s read timeout killed the first attempt on every scheduled run,
  since nothing keeps the free instance warm between the 3-hourly runs.
- **Silence or tune it:** `gh workflow disable watchdog.yml` stops it (`gh workflow enable watchdog.yml`
  restarts); `gh workflow run watchdog.yml --ref main` runs it once out of band. Check by hand without
  writing anything:

  ```bash
  .venv/bin/python scripts/watchdog.py --dry-run                        # verdict + table, no writes
  .venv/bin/python scripts/watchdog.py --dry-run --json                 # machine-readable
  .venv/bin/python scripts/watchdog.py --dry-run --repo jomanor/weather-prediction-mlops  # + planned action
  ```

  `--api` points it at another base (default `https://weather-prediction-mlops.onrender.com`), `--token`
  defaults to `$GITHUB_TOKEN`. Close the issue manually to acknowledge it; a still-bad pipeline recreates
  it on the next run, and so does an API that flaps down→up→down (only *open* issues are matched, so a
  flapping API can open a fresh issue per cycle — up to 8/day at this cron). To change the sensitivity,
  edit the threshold block in `scripts/watchdog.py` and the backend together.

**Frontend — `X-Data-Age-Seconds` is now consumed (Resumen only)**

- `frontend/src/api/client.ts`: the fetch+schema body moved to a private `getJson()` that keeps the
  `Response`; `apiGet` keeps its exact signature/behaviour, and the additive `apiGetWithDataAge()` returns
  `{data, dataAgeSeconds}`. `parseDataAgeSeconds()` accepts only a trimmed `/^\d+$/` that is a safe
  integer — `'-5'`, `'1.5'`, `'12s'` and `''` all yield `null`.
- `frontend/src/api/queries.ts`: `useApiQueryWithDataAge()` (same key/path/schema/options as
  `useApiQuery`) also returns `ageCapturedAt` = React Query's `dataUpdatedAt`, i.e. the `Date.now()`
  baseline for ageing the value forward. No new timer.
- **Precedence / invariant** in `frontend/src/hooks/useFreshness.ts` (`effectiveAgeSeconds`, optional third
  arg `serverAge`):
  1. `carried = serverAge + max(0, round((now - receivedAt)/1000))`, recomputed on the existing 60 s tick,
     then `ageSeconds = payloadAge === null ? carried : max(carried, payloadAge)`.
  2. **The header is a floor, not live authority.** `/weather/current` is served `public, max-age=30` with
     a strong ETag, so a revalidation answers 304 — invisible to `fetch` — and the browser replays the
     stored body *and* the stored header while React Query resets `dataUpdatedAt`. `carried` therefore
     restarts from a value measured at the last real 200 and can freeze while the data ages. `max()` means
     the header can only **raise** the estimate, never lower it, so a stale payload is never painted
     fresh; the value the badge shows is never below the pre-header payload math.
  3. no usable server age (absent or unparseable header, proxy-stripped, old backend) ⇒ `ageInSeconds(observed_at)`
     exactly as before; a server age with no payload timestamp ⇒ `carried` alone (unreachable in practice:
     `observed_at` is required and the API derives the header from the same field).
  4. `isFresh = ageSeconds !== null && ageSeconds <= FRESH_WINDOW_SECONDS` (`12 * 60 * 60`, unchanged), and
     the stale line keys on that result: `stale = ageSeconds !== null && !isFresh`, so "… · datos con más
     de 12 h" also appears when the age came from the server and `observed_at` is missing.
- **Why read the header at all:** it is measured where the data lives, so a client clock *behind* true time
  can no longer make stale data look fresh (the carried server age raises the estimate past it). A clock
  *ahead* can still show fresh data as stale — that direction is bounded, pre-existing, and the safe one.
  The browser can only read the header because the API exposes it in `access-control-expose-headers`
  (verified live: `ETag, Cache-Control, X-Data-Age-Seconds, X-Request-Id`); a proxy that drops it just falls
  back to case 3.
- Wired only for `useStations` (`/weather/current`) in `frontend/src/features/overview/queries.ts` +
  `OverviewPage.tsx`. The rendered line ("… · datos con más de 12 h") and the dot are unchanged; the
  station hero, `StationTable` and the rest of the ~all `apiGet` call sites were deliberately left alone.
- **Closed review blocker (2026-10-31, does not need action):** the first cut let a 304 replay lower the
  reported age and paint stalled data fresh. Fixed by the `max()` invariant above, not by hiding it; a test
  (`'survives a 304 replay storm without freezing the age'` in
  `frontend/src/hooks/useFreshness.test.ts`) pins it. Residual understatement left: only a client clock *behind*
  combined with a frozen header, bounded by exactly what the code did before this feature.
- **Gates measured here** (passes say nothing about correctness): `pytest tests/unit/test_watchdog.py -q`
  → `93 passed`; `scripts/watchdog.py --dry-run` → `veredicto: ok`, exit 0 (3.0 h / 0.994 / 5.2 h /
  1.000 at 11:12 UTC — the values move with the data, plus the two skill advisories); `ruff`/`black
  --check` clean on both new files; frontend 48 files / 351 tests passed, typecheck clean, lint 0 errors,
  build ok. The repo-wide `ruff`/`black` failures (`tests/unit/fake_registry.py`,
  `tests/unit/test_promote_model.py`) and the 60 PySpark `Py4JJavaError` failures in the local root suite
  are pre-existing, untouched by this slice. The earlier gap is closed: `parse_timestamp` now has
  parametrised tests for naive input, `Z`/`z`, fractional seconds, non-UTC offsets and garbage.
- **Review verdict: CLOSED** (re-verified on the fixed tree). Open NITs only:
  `python scripts/watchdog.py --nope` exits `2` while the module docstring still says `1`; the code's
  900 s worst-case model counts 4 GitHub calls where `GitHubClient.comments()` can issue 5 (needs a
  hanging GitHub plus >100 comments).

## 2026-10-31 — Batch 6c: read-API metrics, /series on the station page, k6 cold burst

Shipped 2026-10-31 on `main` in `3049b36` → `81c2682` and **deployed**. New files:
`backend/app/core/metrics.py`, `backend/app/routers/metrics.py`, `backend/tests/test_metrics.py`,
`scripts/load/cold-burst.js`, `.github/workflows/loadtest.yml`,
`frontend/src/features/station/StationPageRequests.test.tsx`. The API half is live — `GET /api/v1/metrics`
answers **200 on production** (checked after the deploy) and the load workflow has been dispatched once and
is **green** (run `41200504597`). The frontend half is committed but not published: auto-publish is paused
(2026-10-31 entry), so the station page changes only when a Netlify deploy is triggered deliberately.
**Review verdict: CLOSED** — the first pass found two defects (unbounded method cardinality
in the metrics table, unthresholded `cold_start_ok`); both are fixed and re-measured, see *fixed* below.

**`GET /metrics` — contract (`backend/app/core/metrics.py`, `backend/app/routers/metrics.py`)**

- Public, unauthenticated, JSON, `Cache-Control: no-store` (no `ETag`, no `X-Data-Age-Seconds`: a replayed
  verdict is worse than none). No new dependency, no exporter, no scraper, no collection, no background
  task; the counters live in `app.state.metrics`, one instance per app. Mounted on the aggregate router in
  `backend/app/main.py`, so `/api/v1/metrics` and the deprecated `/api/metrics` both answer. Full field
  reference in `docs/api-contract.md` ("Metrics"); this section records the limits and what they mean.
- Shape: `started_at`, `uptime_seconds`, `requests{total, by_status_class, errors_5xx, cache_hits,
  cache_hit_rate}`, `routes[{method, route, count, error_count, cache_hit_rate, p50_ms, p95_ms, p99_ms,
  max_ms}]` sorted count-desc, `slo{latency_p95_ms_target, p95_ms, met}`. `error_count`/`errors_5xx` are
  **5xx only** — a 4xx is a client error, not an availability failure.
- **Bounded buckets, not samples.** Latency is counted into 12 counters per route: 11 finite edges
  (5/10/25/50/100/250/500/1000/2500/5000/10000 ms) plus `+Inf`, so memory is O(routes × 12) and cannot grow
  with traffic. A list of durations would grow without bound on the 512 MB instance.
- **Percentile resolution is the bucket edge, no interpolation.** A reported `p95_ms` is the upper bound of
  the first bucket whose cumulative count covers the rank: 30 ms reports 50, 120 ms reports 250, and a rank
  in `+Inf` reports that route's `max_ms`. The estimate is never below the true value, which is what an SLO
  verdict wants; it is not a number to quote as "the p95 was 121 ms".
- **Route templates only.** The key is `(normalized method, route template)` — `/api/v1/weather/history/{city}`, never
  the served path — so no city name, query string, request id or client address reaches the payload. The
  mount prefix is resolved from the served path, not from the route object: FastAPI 0.138.1 shares one
  `APIRoute` across both mounts, so `scope["route"].path` is *unprefixed*. Consequence: `/api/v1/cities`
  and `/api/cities` are separate entries. A 404 raised before routing (and a CORS preflight) is
  `<unmatched>`; a routed failure keeps its template.
- **Per process, resets on restart — and when the free instance sleeps.** These numbers describe the
  current instance's life, never a 28-day window; do not report them as one. A durable window needs an
  exporter or a store (deliberately out of scope). `/metrics` is counted like every other route, so it
  appears from the next call on.
- **How to read it** (no scraper; it is a `curl | jq` surface, live now):

  ```bash
  curl -s https://weather-prediction-mlops.onrender.com/api/v1/metrics | jq -c '.slo, (.requests | {total, errors_5xx}), (.routes[:5] | map({route, p95_ms, max_ms}))'
  ```

  Run today against production: `{"latency_p95_ms_target":2000,"p95_ms":100.0,"met":true}` /
  `{"total":4308,"errors_5xx":0}` / the analytics routes, `/api/v1/health` and `/api/v1/metrics`.
  `slo.met` is `p95_ms <= 2000` for the whole process, and `true` while no request has been recorded
  (nothing has violated it yet).
- **A `p95_ms` above `max_ms` is the convention, not a bug — do not file it.** No interpolation: the value is
  the bucket upper bound for the quantile's rank, so on a low-count route it can sit above every sample that
  route saw. Live sample today: `/api/weather/current` with 1 request reported `p95_ms: 2500.0` against
  `max_ms: 1143.1`, and `/api/cities` reported `p95_ms: 250.0` against `max_ms: 187.0`. Read `max_ms` when
  you want the true tail; treat `p50/p95/p99` as upper bounds.
- **Fixed (blocker): bounded for any client input.** Method is normalized before keying — `KNOWN_METHODS`
  is `GET`/`HEAD`/`POST`/`PUT`/`PATCH`/`DELETE`/`OPTIONS` and `normalize_method()`
  (`backend/app/core/metrics.py:98`) keeps a token only on **exact** uppercase membership, so lowercase
  `get`, `TRACE`, a crafted `M4711` and the empty token all report `<other>` and no client text can reach
  the payload. The key is `(normalize_method(method), route)`; `MAX_ROUTES = 256` and the cap now covers
  **every** row including `<unmatched>` and `<other>`, with one slot always kept free for the single sink
  row `SINK_KEY = ("<other>", "<other>")`: at most 255 real rows + 1 sink, whatever is sent. `MAX_ROUTES`
  moved 64 → 256 because the cap now counts method×route combinations (the app declares ~54 mount+path
  combinations); `test_default_cap_leaves_room_for_every_documented_route` trips if real traffic ever has
  to compete for slots. **Do not go back to a per-path cap with a raw method key** — that is exactly the
  hole this closed.
- **Fixed, measured** (real `create_app()` over ASGI, before → after): 20 000 crafted methods at
  `/api/v1/nope` → **3 entries / 684 B** (was 20 002 entries / 2 809 439 B, +9.5 MB RSS); the RSS delta after
  the fix (+980 kB) equals the control of 20 000 identical `GET`s (+992 kB), i.e. per-request churn, not
  table growth; crafted tokens present **0/3**. Same probe on `/api/v1/health` → 3 entries / 687 B.
  Exhaustive pressure (8 method labels × 500 templates = 4 000 candidates) → **256 entries, one sink row,
  `_admitted = 255`**; +10 000 repeats → still 256. `max_routes=2` with reserved labels → 2; `max_routes=0`
  → 1. Real uvicorn on `weather_db`: 500 crafted methods + 2 real `GET`s → 4 entries / 845 B, with
  `("<other>", "<unmatched>") = 500`. Re-checked here with the recorder alone: 20 000 crafted methods → 1
  entry; 700 real routes + 8 method labels × 500 templates → 256 entries, the sink row
  `("<other>", "<other>")` absorbing 4 445 of them.

**Station page — one request instead of two**

- `frontend/src/features/station/queries.ts`: `useStationSeries(city, hours)` calls
  `/weather/series?cities=<city>&hours=<hours>&fields=<STATION_SERIES_FIELDS>` and replaces both
  `useCurrentWeather` and `useHistory` (`queryKeys.station`/`queryKeys.history` collapse into
  `queryKeys.series`). `StationPage.tsx` renders one payload twice: hero = `points.at(-1)`, chart = the
  ordered points, so the two cannot disagree about the newest observation. Polling stays at 5 min (the
  faster of the two removed queries).
- `STATION_SERIES_FIELDS` is exactly the 10 columns the surface reads (hero: `temperature`,
  `apparent_temperature`, `humidity`, `pressure`, `wind_speed`, `wind_direction`, `precipitation`,
  `cloud_cover`, `weather_code`, `is_day`; the chart adds nothing). That union *is* the whole
  `WEATHER_FIELDS` contract today, so the projection saves 0 bytes — it is pinned so an extra column cannot
  start arriving in a payload nobody reads. `step_hours` is left at the backend default (1 h, no
  downsampling), which is what keeps point parity with `/history`.
- **Measured point parity** (live API, 15 cities × 24 h and × 48 h, plus Madrid 168 h): `series.count ==
  history.count` and `points.at(-1) == /current.observed_at` in every case, mismatches = none. Madrid 24 h:
  27 points both, newest `2026-10-31T08:15:00Z` in all three endpoints.
- **Bytes**: Madrid 24 h `282 + 7915 = 8197` → **7980** (−217 B, 27 pts); 48 h `282 + 14924 = 15206` →
  **14989** (51 pts); 168 h `282 + 55668 = 55950` → **55733** (190 pts).
- **Requests**: 2 observation calls (`/weather/current/{city}` + `/weather/history/{city}?hours=&limit=500`)
  → **1**. `StationPageRequests.test.tsx` pins the recorded paths as exactly `['/cities',
  /^\/weather\/series\?/, /^\/predictions\//]` and asserts `/weather/current` and `/weather/history` are
  never requested, including on the 404/failure path (no silent fallback).
- Verified live today (`curl`, 2026-10-31): the exact `fields=` list → `200`, 7980 B, 27 points, all 10 keys
  present; `fields=temperature,bogus_field` → `422 {"code":"validation_error","message":"Unknown
  field(s): bogus_field"}`:

  ```bash
  curl -s "https://weather-prediction-mlops.onrender.com/api/v1/weather/series?cities=Madrid&hours=24&fields=temperature,apparent_temperature,humidity,pressure,wind_speed,wind_direction,precipitation,cloud_cover,weather_code,is_day" | jq -c '.cities[0] | {count, latest_timestamp, points: (.points|length)}'
  # {"count":27,"latest_timestamp":"2026-10-31T08:15:00Z","points":27}
  ```
- **Behaviour change, inherent to one request**: `/series` answers 200 with an empty city (no per-city 404),
  so a station with no row inside the selected window renders the new "Sin observaciones recientes" empty
  state instead of `/current`'s out-of-window hero. Not reachable on live data (all 15 stations have rows
  < 24 h old). `historyResponseSchema` stays in `frontend/src/api/schemas.ts` as the still-live `/history`
  contract; `useStats` is still an unused export (pre-existing).
- Deliberately skipped: `/summary` (temperature-only sparkline, the hero would still need a second request)
  and `/range` (per-city keyset paging = more round trips for one city). `X-Data-Age-Seconds` untouched —
  `/series` does not set it and the station surface never read it.

**k6 cold burst — `scripts/load/cold-burst.js` + `.github/workflows/loadtest.yml`**

- Per iteration one `http.batch` of the three cached analytics views the dashboard opens with plus
  `/api/v1/health`: `analytics/daily?city=<CITY>&days=<DAYS>` (`city` is required), `analytics/diurnal`,
  `analytics/correlation`, `health`. Defaults 3 VUs / 45 s `constant-vus`, `gracefulStop: 120s`, 60 s
  per-request timeout, **no retries** (a cold failure must be recorded, not papered over), bodies discarded.
  Env overrides `BASE_URL`, `VUS`, `DURATION`, `CITY`, `DAYS`.
- Thresholds: `analytics_ms: p(95)<2000`, `health_failed: rate<0.01`, `http_req_failed: rate<0.05`,
  **`cold_start_ok: rate>0`**. The cold samples are **not** exempt from `analytics_ms` — the budget is the
  budget; `cold_start_ms` (slowest of VU 1 / iter 0's four responses) stays diagnostic, while `cold_start_ok`
  (all four probe responses 200) is a gate, because four 60 s timeouts on a sleeping instance are ~4 requests
  in ~2400 (~0.2 %) and stay silent inside all three rate/latency budgets.
- **Dispatch-only, never on push or a schedule**: the target is a free Render instance that sleeps after
  ~15 min idle, so a scheduled run would be a load test against production on a timer, and RED there means
  the API missed the budget, not that CI broke. `workflow_dispatch` inputs `base_url` / `vus` / `duration`,
  `permissions: contents: read`, k6 pinned `2.3.0` via `grafana/setup-k6-action@v1` (no leading `v`, see
  below), `--summary-export=summary.json` uploaded as the `k6-summary` artifact `if: always()` (14 days).
  Statics: `yaml.safe_load` → triggers `['workflow_dispatch']`, `permissions {'contents': 'read'}`, 4 steps;
  `node --check` on the script passes; `k6 inspect` resolves the four thresholds.
- **Observed runs (production, 2026-10-31)** — no k6 installed system-wide, release tarballs unpacked into
  `/tmp`, all exit **0**, no threshold crossed:

  | k6 | requests / iters | req/s | `analytics_ms` p95 (max) | `cold_start_ms` | `cold_start_ok` | `health_failed` | `http_req_failed` |
  |---|---|---|---|---|---|---|---|
  | v0.54.0 (pre-gate) | 2352 / 588 | 52.2 | 222.19 ms (4450.10) | 4445.07 ms | 1/1 | 0/588 | 0/2352 |
  | v2.3.0 (pre-gate) | 2552 / 638 | 56.5 | 225.41 ms (577.94) | 237.73 ms | 1/1 | 0/638 | 0/2552 |
  | v2.3.0 (with the gate) | 2360 / 590 | 51.98 | 229.52 ms (4340) | 4340 ms | 1/1 | 0/590 | 0/2360 |

- Gates proven live, each on a real run (exit **99**): `analytics_ms` via a tuned `p(95)<1` copy;
  `cold_start_ok` via the repo script against a stalled TCP listener (`VUS=1`, `DURATION=1s`) → all four
  probes `1m0s`, `cold_start_ok 0/1`, and the cold samples still inside `analytics_ms` (p95 `1m0s`) and
  `http_req_failed` (4/4); and against `BASE_URL=http://127.0.0.1:9 VUS=1 DURATION=2s` → exit **99**,
  `thresholds on metrics 'cold_start_ok, health_failed, http_req_failed' have been crossed`. Isolation
  checked on `/tmp` copies of the script: only `cold_start_ok` left → exit **99**; gate removed → exit
  **0** while `cold_start_ok` still read `0.00% 0 out of 1` — which is the silent green the gate exists to
  kill. `VUS=0` → exit `104` (k6 rejects the run, so there is no "no sample, still green" hole either).
  `--summary-export` is written even when a gate crosses, and its per-threshold boolean reports **crossed**
  — `false` in `summary.json` means the run passed, not the reverse.
- **Dispatched 2026-10-31 and green.** Run `41200504597` on `main` (`81c2682`), 3 VUs / 45 s, every job
  step success including the `k6-summary` artifact upload: **4300 requests / 1075 iterations / 95.2 req/s**;
  `analytics_ms` p95 **181.97 ms** (avg 112.99, med 97.43, **max 4.89 s** — the cold hit); `cold_start_ms`
  **4.89 s** with `cold_start_ok` **1/1**; `health_failed` 0/1075; `http_req_failed` 0/4300; checks
  4300/4300; `summary.json` attached as `k6-summary`. Read a run with (both run here):

  ```bash
  gh run list --workflow=loadtest.yml --limit 1
  gh run view 41200504597 --log | grep -E 'analytics_ms\.|cold_start_ms|cold_start_ok|health_failed|http_req_failed|http_reqs'
  ```

- **The first dispatch failed for a CI reason — keep the k6 version without a leading `v`.** Run
  `41200392533` (`15cf013`) died in `grafana/setup-k6-action@v1` with `Unexpected HTTP response: 404`: the
  action prefixes the release tag itself, so `k6-version: v2.3.0` asks for
  `…/releases/download/vv2.3.0/k6-vv2.3.0-linux-amd64.tar.gz`. Fixed in `81c2682` to `k6-version: 2.3.0`,
  with that reason in a comment in `.github/workflows/loadtest.yml`.
- **CI on `main` had been red since the Batch 6b watchdog landed, for a test-only reason.**
  `tests/unit/test_watchdog.py::test_missing_repo_exits_1` passed locally only because no `GITHUB_*`
  variable was set there; on Actions `GITHUB_REPOSITORY` is ambient, so `run()` resolved a repo instead of
  exiting `1`. Fixed in `15cf013`: an autouse fixture (`_hermetic_env`, over `ENV_DERIVED_DEFAULTS`) clears
  `GITHUB_REPOSITORY`/`GITHUB_TOKEN` for every watchdog test, plus a regression test pinning both the cleared
  and the Actions-shaped environment — the suite is 94 tests. CI run `41200090999` is green end to end,
  including **Deploy (Render)**.
- **Fixed (should-fix): the cold probe is now a gate.** `cold_start_ok: ['rate>0']` in
  `scripts/load/cold-burst.js`; one sample per run, so `0/1` fails the run where the three rate thresholds
  and the p95 could not see it. The `loadtest.yml` RED-condition comment names it. It is **not** an
  exemption: the cold samples still count in `analytics_ms` and `http_req_failed`.

**What remains unmeasured**

- **A real Render boot (30–60 s) is still unseen.** Every run so far hit an already-warm instance —
  `cold_start_ms` 4445.07 / 237.73 / 4340 / 4890 ms — including the dispatched one, which started a few
  minutes after the deploy finished. The 60 s timeout branch itself is exercised (stalled TCP listener →
  all four probes `1m0s`, exit 99), so the mechanism is proven; what is unproven is how long a genuinely
  sleeping Render instance takes to answer, and by extension whether 60 s is the right timeout. To measure
  it, dispatch after the instance has idled ~15 min — and note the 4.89 s cold hit above already sits inside
  the `analytics_ms` p95 as designed.
- **One dispatch is a sample, not a trend.** A single 45 s run at 3 VUs says nothing about drift, and
  nothing runs this on a cadence (deliberately: see *dispatch-only*). Compare runs by re-dispatching, not by
  reading a time series.
- **No SLO exists, only a snapshot.** `/metrics` reports `p95_ms` against a literal 2000 ms target, but an SLO
  needs an agreed target and a reporting habit, and it has neither: the counters are per process and reset
  when the free instance sleeps, nothing reads them on a schedule, and no window is defined. Needs a durable
  window (exporter or an agreed sampling rule over `/metrics`), a signed-off target, and an owner/cadence.
- **The frontend was not seen in a browser.** No harness in this repo: the chart, zoom/pan and the 5-min
  polling were not visually confirmed. `HistoryChart` is unchanged and receives the same point shape/order
  (its own tests cover ordering and zoom).
- **`/series` has no `limit`**: a multi-city 168 h request is unbounded (parity with `/history`, not a
  regression) — that is the future truncation axis and the backend owner's call.
- **Frontend gates** (passes say nothing about correctness): 49 files / 370 tests (was 351), typecheck clean,
  lint 0 errors (one pre-existing `vite.config.ts` warning), build ok, `StationPage-*.js` 20.57 kB / 7.26 kB
  gzip. Backend: `312 passed`, `ruff`/`black --check` clean (67 files unchanged); `backend/tests/test_metrics.py`
  covers method normalisation, the sink row, the cap and the hostile-input probes.

## Next: Batch 6

No scope frozen yet. The cheapest remaining items are in `docs/roadmap.md` under *Accepted follow-ups* —
read that list, not this paragraph. Two of them moved in Batch 6c: the station page now consumes `/series`
(`/summary`, `/range` and `/api/map/stations` are untouched), and the k6 cold-burst workflow has now been
dispatched once and is green — what is left there is a cold-instance measurement, not a harness. The read-API
SLO still does not exist: `/metrics` gives a per-instance snapshot, not the agreed target + reporting habit
an SLO needs. The Resumen freshness line is guarded by the `max()`
invariant above; it understates only for a client clock behind true time. Browser verification still needs
the desktop app session.

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
- Healthchecks.io checks for the ingest/inference/train crons (optional now: the repo-local `watchdog.yml` from Batch 6b covers ingest + features + inference staleness, so Healthchecks.io would only add an out-of-band liveness ping; the in-repo `::warning` is ephemeral).
- Atlas project alerts (disk >80%, connections >80%, query targeting).
- Browser verification needs this session open in the desktop app, otherwise the browser tools report `disconnected`.

## 2026-10-31 — Batch A: the writers tell the truth (audit findings 1, 2, 6, 12, S4, S5)

Committed as **`df39022`** on `main` (batch B sits on top as `cfabdf3`).
Files: `scripts/backfill_historical_data.py`,
`tests/unit/test_backfill.py`, `scripts/requirements.txt` (new), `backend/app/repositories/weather_repo.py`
(one docstring), `.github/workflows/{ingest,backfill,ml-pipeline,ci}.yml`. No backend/runtime code path
changed; `spark/` untouched. Reviewer verdict **CLOSED** after three passes (blocker + should-fix fixed,
listed below); no database-engineer or sre gate applies (no index, TTL, schema or SLO change).

- **F1 — a failed write is no longer a "skip".** `upsert_batch` treats *only* BulkWriteError entries with
  code 11000 as duplicates (like `$setOnInsert` intend); any other code, a mixed batch, or a
  `BulkWriteError` with no `writeErrors` raises `RuntimeError` chained from the original. `main` now splits
  `api_failures` (fetch returned None, or a 200 whose response yields zero hourly docs, in both the archive
  and the recent-fill loops) from `write_failures` (that RuntimeError, or inserted+skipped != len(docs)),
  aborts on a write failure, always closes the client, always prints the summary, and exits **1** for any
  write failure and — by default — for any API failure. "Backfill complete." is only printed on a clean
  run, so a green `backfill.yml` run now means rows were actually written.
- **F2 — the default range fits the tier.** Default is the last **90 days** (`DEFAULT_DAYS`, computed with
  `today - timedelta(days=90)`, so the old `date(year-4, 2, 29)` leap-day crash is gone), `MAX_RANGE_DAYS`
  = **180** refuses anything longer (the `raw_weather` TTL would delete it on arrival) with exit 1 *before*
  any `MongoClient` is built, and `--chunk-days` is validated (`>= 1`).
- **F6 — the local hour is DST-correct at the writer.** `build_documents` no longer stamps one response's
  `utc_offset_seconds` on a multi-month chunk: it recomputes the offset **per document** from
  `payload["timezone"]` (IANA) and that row's own timestamp via stdlib `zoneinfo`, with a per-name cache and
  a never-raising fallback chain (payload offset, then `None`). `spark/spark-jobs/batch_processing.py` was
  deliberately NOT changed: `_local_timestamp_col` already prefers the stored offset and falls back to its
  longitude solar offset, so fixing the writer is the whole fix. `tzdata` is now declared in
  `scripts/requirements.txt` — without an IANA database the fix would silently revert.
- **F12 — behaviour tests, not source greps.** `tests/unit/test_backfill.py` went 20 → **68** tests: real
  calls to `upsert_batch`/`upsert_documents` (duplicate vs real write error, mixed, empty `writeErrors`,
  batching), the DST regression, `resolve_range`, the exit matrix, the per-city empty-response case, the
  recent-fill write path (success + future-row filter + RuntimeError + mismatch), and the dry-run
  Mongo-unreachable fallback. Source-text assertions (`"$setOnInsert" in source`) are gone. Suite is
  TZ-independent (passes under `TZ=UTC` and `TZ=Pacific/Auckland`) and offline; QA mutations on five
  boundaries each killed at least one test.
- **S4** `already_exists` deleted; the module docstring's Idempotency/Defaults/Options sections rewritten to
  match the code; `weather_repo.quality_report`'s "fixed 14-station registry" → `DEFAULT_CITIES` (15, incl.
  Pasadena).
- **S5** new `scripts/requirements.txt` (`requests`, `pymongo`, `python-dotenv`, `tzdata`) installed by
  `ingest.yml`, `backfill.yml`, `ml-pipeline.yml` and both `ci.yml` jobs; no workflow hand-installs those
  any more. Still compatible with `backend/requirements.txt` (pymongo==4.17.0) and `spark/requirements.txt`
  in the same job.
- **`--dry-run`** (added because the audit's gate asks for a 90-day dry run) validates the range, resolves
  the station set best-effort (`serverSelectionTimeoutMS=750`, falls back to the bundled list when Mongo is
  unreachable) and prints the plan, document and chunk counts — no Open-Meteo call, no write.
- **`--best-effort` + `continue-on-error: true` on ml-pipeline's `--days 2 --recent 2` step.** This was the
  reviewer's blocker: with the new strict exit, one archive hiccup would have skipped the feature rebuild,
  training and inference (the step is a density top-up, not a correctness gate). Under `--best-effort` an
  API failure becomes `::warning title=Backfill partial::` and exit 0; a write failure still exits 1 and now
  also emits `::error title=Backfill write failure::`. The repair run in `backfill.yml` keeps the strict
  default on purpose.
- **Gates measured here:** backend `312 passed`; root `457 passed, 7 skipped` (`PYSPARK_PYTHON=$PWD/.venv/bin/python`);
  frontend 370 tests, typecheck clean, lint 0 errors (one pre-existing `vite.config.ts` warning), build ok;
  `ruff` + `black --check` clean over `backend scripts spark` (83 files); `yaml.safe_load` on the four
  workflows; `--days 90 --dry-run` exit 0, `--days 200 --dry-run` exit 1, `--chunk-days 0` exit 2.
- **Residuals (deliberate, do not "fix" blind):** (1) rows already in `raw_weather` keep their old
  single-stamp offset until the 180-day TTL removes them — `$setOnInsert` never overwrites, so the fix only
  applies to newly backfilled ranges; repairing the stored field would be a data write needing the owner's
  go-ahead. (2) A 200 whose `hourly` series is truncated but non-empty still counts as success (only a
  zero-row response is a failure). (3) The `nothing was written` guard is defence-in-depth: with an honest
  writer, a real failure is already counted by the two counters above. (4) **No retrain was run**: feature
  *semantics* did not change, so hard rule 1 is not triggered yet — it becomes mandatory after the first
  `job=features` that ingests rows from a DST-spanning backfill (the audit's F6 note).

## 2026-10-31 — Batch B: one authority per rule (audit findings 3, 4, 5, 11 + S1)

Committed as **`cfabdf3`**, review fixes **`645ba16`** and **`04ad956`** (**CLOSED**: three review
passes, 4 should-fix items + 1 compose path fixed, none waived). Built in the `batch-b` worktree,
fast-forwarded into `main` because the `/refresh` deletion was in flight in the main tree. `docs/api-contract.md` updated for the changed
threshold, the new cache headers and the additive `/metrics` key.

- **F3 — `raw_weather` is read first, `weather_data` is the fallback.** `latest_for_city`,
  `latest_per_city`, `find_range` and `find_many_in_range` flipped: the live collection is queried first and
  the legacy snapshot only when it has no rows, so one leftover legacy document can no longer replace the
  fourteen live stations or shrink a history to a point. One round-trip fewer on the hot path; sorts stay
  `{city:1, timestamp:-1}`, aggregates keep `allowDiskUse`, output stays oldest→newest. The collection and
  its TTL stay (hard rule 3).
- **F4 — the Mongo `cities` registry is the only runtime city set.** New
  `city_repo.registry_names(db)` (one `find({}, {name:1})`, sorted, deduped, `DEFAULT_CITIES` only when the
  registry is empty/unreadable, never raises) is now read by `quality_report`, the diurnal/correlation/
  error-by-hour aggregates (the module-level `CANONICAL_CITIES` constant is gone) and, for its fallback,
  the backfill script. `DEFAULT_CITIES` is seed data + the empty-registry fallback and nothing else, so a
  station added or deleted through the API is ingested *and* monitored; the backfill's separate 14-station
  dict (no Pasadena) is gone.
- **F5 — `backend/app/core/thresholds.py` is the single authority.** Plain constants, no imports, imported
  by the API (`weather_repo`, `routers/predictions`), `spark/spark-jobs/inference.py`,
  `scripts/pipeline_summary.py` and the stdlib-only `scripts/watchdog.py` (they add `<repo>/backend` to
  `sys.path`; the module drags in no framework). **Prediction source age is 12 h everywhere** — the API's
  `stale` flag previously said 24 h while the watchdog said 12 h. No env override: `config.prediction_stale_after_hours`
  was deleted on purpose. The watchdog/watchdog-docstring "mirror these numbers" instructions are gone.
- **F11 — index failure is visible; the two heaviest reads are cached.**
  `_prepare_database` records `{"ok", "error"}` on `app.state.index_status` and `GET /metrics` reports it as
  the additive `database.indexes` key (`ok: null` before the background task runs). Deliberately *not* in
  `/health/ready`: a missing index is slow, not unservable, and readiness failure would make Render restart
  a working process. `/predictions/latest` (30 s, per-horizon key) and `/weather/series` (30 s, key =
  cities+hours+sorted fields+step) are now cached with ETag/`Cache-Control`/304 — verified live: 200 with
  ETag, then 304 on `If-None-Match`. `TTLCache.delete(key)` drops both the value and the in-flight load
  registration (a loader only caches while it is still registered), and `POST`/`DELETE /cities` call it
  after a successful write only.
- **S1 — one env file.** `.env` + `.env.production` + `atlas-credentials.env` are collapsed into a single
  repo-root `.env` (mode 600, gitignored): `MONGO_DB` (was `MONGO_DATABASE`), the Atlas `MONGO_URI` so
  local runs show real data, `MONGO_USERNAME`/`MONGO_PASSWORD` (docker-compose mongo only) and
  `AEMET_API_KEY`; the local-mongo alternative is a commented `MONGO_URI`. Deleted keys —
  `WEATHER_API_KEY`, `KAFKA_BROKER`, `API_URL`, `MONGO_DATABASE`, `MONGODB_*` — and `NETLIFY_AUTH_TOKEN`,
  which the repo never read, are in `.env.dropped-keys` (mode 600) so a human-held secret is not lost:
  **move that token to wherever your netlify CLI expects it, then delete the file.** It was never
  committed (`git log -- .env*` is empty), so no rotation is needed. `Settings` now reads
  `env_file=(".env", "../.env")`, so `cd backend && uvicorn app.main:app` also picks the file up;
  `docker-compose.yml` uses `${MONGO_DB}`; `.env.example` is tracked (gitignore got a `!.env.example`
  negation) and is the onboarding placeholder. Production config is not a file: `render.yaml` + Render
  dashboard, Netlify UI for the frontend.
- **Also on `main`, not this batch:** `decaba5` deleted `POST /refresh` and its limiter (another session).
- **Gates measured on the merged tree:** backend **333 passed** (the count moved from 312 because the 22
  `/refresh` tests were deleted in `decaba5`); root **461 passed, 7 skipped**; frontend 370 tests,
  typecheck clean, lint 0 errors, build ok; `ruff`/`black --check` clean over `backend scripts spark`
  (81 files). Live, local API against Atlas: `/api/v1/weather/current` → **200, 15 stations incl.
  Pasadena**; `/api/v1/quality?days=7` → 15 cities all `ok`; `/api/v1/metrics` → `database.indexes
  {ok: true}`; legacy `/api` alias 200; `/health/ready` 200; `/series` 32 points oldest→newest.
  `scripts/watchdog.py --dry-run` → `veredicto: ok`, exit 0 (1.0 h / 1.000 / 7.8 h / 15-of-15).
- **Deliberate leftovers:** `quality_report`'s completeness gates stay 0.95/0.80 (only the age gates moved
  into the shared module — completeness is that meter's own scale, and the one-shot backfill's 0.98 must
  keep flagging a broken ingest cadence); the registry is read once per aggregate rather than cached (one
  name-only query, and a cache would need invalidation on every registry write); `weather_data` is still
  present with its TTL.

**Batch B review fixes (all applied before closing).** Four should-fix items came out of the
review/QA pass and are fixed, not waived:

1. **The frontend was a fifth judge of prediction age.** `LatestPrediction.tsx` still had its own
   `PREDICTION_STALE_HOURS = 24` for the legacy-payload fallback, so the badge could say "Más de 24 h"
   while the API flagged `stale` at 12 h. Now 12, with two tests: one pins the literal (TypeScript
   cannot import the Python module, so this test *is* the cross-language contract) and one proves a
   13 h legacy payload is flagged.
2. **The Spark image could not import the shared module.** `spark/Dockerfile` copies only `spark-jobs/`
   to `/opt/spark-jobs/`, so the bootstrap resolved to `/backend`, which did not exist →
   `ModuleNotFoundError` at import (and the compose scheduler only logs a subprocess failure). Fixed
   by (a) `docker-compose.yml` bind-mounting `./backend/app` at `/backend/app` read-only for
   `spark-scheduler`, and (b) `inference.py` trying `<repo>/backend` then `/backend` and, if neither
   exists, raising a message that names the rule ("do not restate the numbers here"). The repo layout
   was verified by importing the module from `/tmp`; CI (`ml-pipeline.yml`) is unaffected.
3. **`QUALITY_WARN_COMPLETENESS = 0.80` was a second literal of the shared
   `QUALITY_BAD_COMPLETENESS_BELOW`.** It is now an alias of the shared constant, and a test asserts
   identity so a retune cannot leave the meter behind. `QUALITY_OK_COMPLETENESS = 0.95` stays local
   (the meter's own scale).
4. **`scripts/sync_to_atlas.py` still read `.env.production`** (reachable via `make sync-atlas`), which
   S1 deleted. It now loads the single `.env`, takes the Atlas target from `MONGO_URI` (or
   `ATLAS_MONGO_URI`) and — important after the unification — builds the *source* from
   `MONGO_USERNAME`/`MONGO_PASSWORD` or `LOCAL_MONGO_URI`, because `MONGO_URI` now points at Atlas and
   the old code would have synced Atlas onto itself. Its usage text is updated too.

QA added the missing test for the `_run_loader` `finally` identity guard (the one surviving mutant: an
older load finishing used to be able to unregister a newer in-flight load, so the newer value was never
cached) — it fails when that guard is removed. `registry_names` now has a malformed-row test, and the
test double no longer filters out the very documents the guard exists for. One unexplained
`test_cache.py` flake was seen once by the reviewer in ~90 runs and could not be reproduced in 2000
targeted replays; the invariant it asserts is now covered by two deterministic tests.

## 2026-11-01 — Batch C: kill the dead weight (audit 13, 14, 17, 18, 19, 20 + S2, S6, S9, S11, S12)

**Local, uncommitted.** No commit id yet, nothing pushed, **no Netlify deployment** — auto publishing
stays paused (`netlify.toml` `ignore = "exit 0"`), so the state of this batch is *committed to the
working tree, not deployed*. No `gh workflow run` was issued. Reviewer/QA verdicts for this batch are
**not recorded here**; only the gates below were measured, by the docs pass, on this tree.

- **F13 — the four endpoints with no client are gone.** Deleted `GET /api/weather/range/{city}`,
  `GET /api/weather/summary` and `GET /api/weather/stats/{city}` (and their `/api` aliases, which follow
  the double mount) together with the schemas `CitySummary`, `SummaryResponse`, `WeatherRangeResponse`,
  `Stats`, `WeatherStats` and `backend/app/services/stats.py` (+ `backend/tests/test_stats.py`). Surviving
  weather routes: `/current`, `/current/{city}`, `/history/{city}`, `/series`, `/quality` — `/series` is
  live on the station page since Batch 6c, so it stayed. `/metrics`, `/map/stations`, `/cities`,
  `/health`, `/models`, `/predictions/*`, `/benchmark/*` and all six `/analytics/*` routes are untouched.
- **F14 / S12 — the frontend's optimistic self-image is gone.** Deleted
  `components/ui/Tabs.tsx` (no consumer; Sistema uses `Segmented`), `lib/derived.ts` (+ its test),
  `components/shell/CommandPalette.tsx`, `command-palette.ts` and `shortcuts.ts`, and the hooks
  `useHealth`, `useLatestPredictions`, `useStats`, `useAnalyticsClimatology`, `useAnalyticsCorrelation`,
  `useAnalyticsErrorByHour` with their schemas (`healthSchema`, `latestPredictionsSchema`,
  `statsResponseSchema`, the geo schemas, and the climatology/correlation/errorByHour blocks).
  `useAnalyticsDaily`/`Diurnal`/`WindRose` stay (they render).
- **S11 — the palette and the `g` chords are gone.** `/` and Cmd/Ctrl-K now focus the single
  `StationSearch` (`StationSearch.tsx` binds both); `AppShell` mounts one search — inline from `md` up,
  its own row below it — so there is one element id and one set of global listeners. The `g r`/`g e`/`g s`
  chords are deleted (the Batches-4/5 entries above are marked accordingly, not rewritten).
- **F17 — one precision.** `PRECISION` in `lib/format.ts` is the only source of decimal places
  (temperature/wind/precip/error/skill); 7 dead helpers deleted (`formatDegrees`, `formatWithUnit`,
  `formatSignedWithUnit`, `normalise`, `formatLatitude`, `formatLongitude`, `formatTemperatureWithUnit`).
  The audit's "nine" was rounded up.
- **F19 — the a11y slips.** Benchmark rows now sit in a `role="grid"`/`role="row"` structure so
  `aria-selected` is announced, the map label is no longer an `aria-label` on a plain `<div>`, the third
  theme option is named "Tema del sistema" (not "Sistema"), and the doubled `StationSearch` is one mount.
- **F20 — one authority per small rule.** `backend/app/core/units.py` holds `MS_TO_KMH = 3.6` (the two
  repositories no longer each declare it); `CACHE_TTL_LIVE_SECONDS` (30) and
  `CACHE_TTL_AGGREGATE_SECONDS` (300) live in `core/cache.py` instead of being re-typed per router;
  `scripts/watchdog.py` pages the open-issue scan (`GITHUB_PAGE_SIZE = 100`, `GITHUB_MAX_PAGES = 50`, so
  a repo past 100 open issues stops forking duplicate rolling issues);
  `scripts/dedupe_predictions.py` streams a projection (`_id`, `city`, `source_timestamp`,
  `horizon_hours`, `prediction_timestamp`) and deletes in `$in` batches of 1000 instead of
  materialising the collection. `pyproject.toml` **already had** `[tool.pytest.ini_options] testpaths`
  (`["tests", "backend/tests"]`) — nothing to add. **It does not make bare `pytest` work**, and the
  audit's premise that it would is wrong: both suites are Python packages named `tests`, so collecting
  them in one run raises `ImportPathMismatchError` (`tests.conftest` vs `backend/tests/conftest`);
  `--import-mode=importlib` does not help either (duplicate conftest plugin name). The two supported
  entry points stay the per-directory invocations in `Makefile` and CI — do not "fix" this by renaming
  the packages without re-checking `tests/integration/*`, which imports `from tests.conftest import
  REPO_ROOT`.
- **S2 / S9 — Spanish UI, Spanish labels.** `ModelsPanel` maps the registry stage through
  `stageLabel` (`production` → "Producción", `staging` → "Staging"); `DataQualityMeter` renders the table
  only when `flagged > 0`.
- **S6 (partial) / CI — one interpreter.** Every workflow is Python **3.13** (`backfill`, `ingest`,
  `watchdog`, both `ci.yml` jobs, the pyspark job included) **except `ml-pipeline.yml`, which stays 3.11**
  because `spark/requirements.txt` pins `pandas==2.0.3` and that release ships only
  cp38–cp311 wheels — the install would fall back to a source build that does not compile. Unpin pandas
  when Batch E drops Spark.
- **Gates measured on this tree** (2026-11-01, after the repo-half deletions): `cd backend &&
  ../.venv/bin/python -m pytest tests -q` → `325 passed`; `PYSPARK_PYTHON=$PWD/.venv/bin/python
  .venv/bin/python -m pytest tests -q` → `468 passed, 7 skipped`; `cd frontend && npm run test` →
  `Test Files 47 passed (47)` / `Tests 350 passed (350)`; `npm run typecheck` clean, `npm run lint`
  0 errors (one pre-existing `vite.config.ts` warning), `npm run build` ok; `ruff check` + `black
  --check backend scripts` clean (72 files).
- **Deliberate leftovers:** `ChartBrush` is **kept** — it is live, mounted by `BenchmarkChart.tsx`, so
  finding 14's "dead primitive" verdict was wrong; the `docker-compose` `mlflow` / `spark-history` /
  `spark-scheduler` services and `spark/Dockerfile` are deferred to Batch E (they go with Spark);
  `historyResponseSchema` has zero consumers but mirrors a live route, so it stays until the
  `openapi-typescript` generation lands (F3); `frontend/src/api/client.test.ts` still uses the literal
  path `/weather/stats/x` as a throw-away URL for the `allowNotFound` cases — it never hits the route;
  the dedupe keeps the same ObjectId list semantics (`_id` added to `to_delete` in the same order).
- **F14 (repo half) / S6 — the phantom dependencies are gone.** Deleted `scripts/sync_to_supabase.py`,
  the `Makefile` `sync-supabase` target and the `netcdf4`/`xarray` declarations in the root
  `pyproject.toml` (`dependencies = []`); `uv lock` dropped `netcdf4`, `xarray`, `pandas`,
  `python-dateutil`, `pytz`, `six` and the lock-file `tzdata`. Neither package was installed in the
  venv (`import netcdf4` already failed), so the declaration was phantom: `pruebas.py` was its only
  importer and that script needs a `.nc` file that never existed. `pruebas.py` itself is **gitignored,
  not part of the repo** (`.gitignore:9`), so it was left on disk instead of silently deleted; it is now
  unrunnable without its two packages, which is the intended end state. `pyspark` stays until Batch E.
  `tzdata` stays declared in `scripts/requirements.txt` (the backfill's zoneinfo needs it) — only the
  root lock lost it, and nothing installed from the lock needs it.
