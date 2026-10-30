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
| Frontend | Netlify site `weather-mlops-1787759329` | Linked to `main` with base dir `frontend`, `npm ci && npm run build`, publish `dist`. `netlify.toml` is the source of truth (base, NODE_VERSION 22, headers, SPA redirects, `/api/*` edge proxy to Render). Auto publishing must stay enabled. |
| DB | Atlas M0 `weather_db` (512 MB) | Re-measured 2026-09-25: `collStats.avgObjSize` `raw_weather` 1362 B, `weather_features` 3856 B, `weather_predictions` **437 B** (Batch-3 interval columns; the old 405 B is stale), live city count **15** (includes Pasadena). Projected steady state **~368.7 MiB** against a 384 MiB (75 %) budget → **~15.3 MiB hard headroom**. Prediction rows are keyed `(city, source_timestamp, horizon)` and written as upsert, so growth follows the ~6 h feature-rebuild cadence, not the hourly inference schedule; the storage-budget test's tripwire is **≥14 rebuilds/day** (8/day is modelled as cadence-jitter headroom). `weather_features` is the largest collection and is a single overwritten snapshot, not an accumulator. The other breach axis is extending the raw TTL — re-measure before changing it. No TTL change in Batch 4: `weather_predictions` stays 90 d. |
| Cron | GitHub Actions | `ingest.yml` declared hourly `:50` (see the live caveat below); `ml-pipeline.yml` inference hourly `:05`, features `20 */6`, train nightly `35 2`; `backfill.yml` manual. |

Deleted branch: `rebuild/v2` is gone (it caused a stale Render deploy). `main` is the
only branch that matters.

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
curl -s https://weather-mlops-1787759329.netlify.app/ | grep -oE 'assets/[A-Za-z0-9_.-]+\.js' | sort -u
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

## State at handoff (2026-09-25, after Batch 4)

- Batches 1–3 are merged and deployed on `main` (Batch 3 = the previous squash commit).
  Batch 4 is **complete but uncommitted** on `feat/batch4` (working tree; do not
  commit/push — the lead integrates). Frozen interfaces: `docs/batch4-contract.md`.
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
  - Pasadena is in Atlas but absent from `FALLBACK_CITIES`/`DEFAULT_CITIES`/`CANONICAL_CITIES`, so several endpoints (incl. diurnal/correlation/error-by-hour) omit it.
  - `weather_features` is overwritten each rebuild, dropping its `{city:1,timestamp:-1}` index while `ensure_indexes` only runs at backend startup; the new analytics endpoints are the first hot consumers → COLLSCAN + in-memory sort until a restart (latency, not code 292, at current volume).
  - `weather_data` still holds 0 rows — expected; the R2 TTL entry for it is vestigial. Leave it in place.
- **Trap fixed in Batch 3, do not regress it:** `inference.py` used to trust the MLflow registry for artifact metadata; when MLflow served the artifact the row lost its `interval` and reported fake `MLflow-latest` lineage. `_latest_model_entry` is now resolved first and always supplies metadata; MLflow only ever serves the *artifact*.

## In flight (2026-09-30) — `fix/spark-atlas-transient-retries` (Spark job resilience)

Worktree `../weather-prediction-mlops-atlas-retry`, branched off `main` a40b87d in its **own worktree** so the
user's uncommitted Batch-4 session in the main checkout is untouched. Committed as `a7bcbae` on
`fix/spark-atlas-transient-retries` (not pushed); worktree clean. Owner: lead
+ `ml-engineer`; interfaces frozen before fan-out.

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

**Not merged, not pushed — the user's call (2026-09-30): keep the branch local.** No security gate (no auth,
secrets or input surface changed). The user merges when the Batch-4 session is committed; pushing later would
run the real `pipeline-test` on Python 3.11 if that extra evidence is ever wanted.

**Agreed follow-up (user approved 2026-09-30):** once Batch 4 is committed in the main checkout, delete the
duplicated retry helper inside `spark/spark-jobs/batch_processing.py` and import `run_with_mongo_retries` from
`mongo_retry` so feature engineering, inference and training share one implementation. Do **not** touch that
file while it is still uncommitted.

## Next: Batch 5

Scope: **M5** stages (`staging` -> human promote -> `production`) + rollback script + shadow A/B, **M6** declare GridFS + `model_registry` canonical (the CI MLflow file store is ephemeral), **L6** optional rate-limited `POST /api/refresh` (server-side fetch; never a browser-triggered workflow dispatch; WebSocket rejected), **R6** structured JSON logs with `request_id`/`duration_ms`/`cache_hit`, stable error envelope, `/api/v1` mount (breaking — do before the frontend hardens). The canonical model store, promotion flow, shadow semantics, fallback rule and env vars are declared in `docs/model-lifecycle.md` (M6).

Also carry the accepted follow-ups in `docs/roadmap.md`: bound/LRU the cache keyed on the unbounded `city` string, k6 cold-burst test, read-API latency/availability SLO (needs an OTel histogram or structured slow-request logs first), production-drift SLI + pipeline dead-man's switch, re-run `ensure_indexes` after the feature write (done in M6 — `batch_processing.ensure_features_index` recreates `{city:1,timestamp:-1}` after every write, non-fatally), the Pasadena registry gap, the `unix_timestamp` `_id` TZ dependency, surfacing `X-Data-Age-Seconds` to the frontend, and the still-unconsumed `/series`/`/summary`/`/range` endpoints.

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
