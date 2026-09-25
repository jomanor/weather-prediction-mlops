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
  - **M4 diagnostics** (additive `model_registry.diagnostics`): `by_city`, `by_hour_of_day` (24, local hour), `by_rain_bucket` (dry <0.1 / light 0.1–2 / moderate 2–8 / heavy ≥8 mm), `drift_psi` (10 equal-width train bins, eps-guarded, null when <10 rows; a train→test proxy). Temperature slices fill rmse/mae/bias, rain slices brier, empty slices `n:0` + nulls. Exposed via `/api/models` (`ModelInfo.diagnostics`) and rendered as a ModelsPage "Deriva" badge (`<0.1` ok / `≤0.25` warn / `>0.25` drift, neutral when absent); `scripts/pipeline_summary.py` emits a non-blocking `::warning` for models with `maxDriftPsi > 0.25`.
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

## Next: Batch 5

Scope: **M5** stages (`staging` -> human promote -> `production`) + rollback script + shadow A/B, **M6** declare GridFS + `model_registry` canonical (the CI MLflow file store is ephemeral), **L6** optional rate-limited `POST /api/refresh` (server-side fetch; never a browser-triggered workflow dispatch; WebSocket rejected), **R6** structured JSON logs with `request_id`/`duration_ms`/`cache_hit`, stable error envelope, `/api/v1` mount (breaking — do before the frontend hardens).

Also carry the accepted follow-ups in `docs/roadmap.md`: bound/LRU the cache keyed on the unbounded `city` string, k6 cold-burst test, read-API latency/availability SLO (needs an OTel histogram or structured slow-request logs first), production-drift SLI + pipeline dead-man's switch, re-run `ensure_indexes` after the feature write (or append/merge-upsert), the Pasadena registry gap, the `unix_timestamp` `_id` TZ dependency, surfacing `X-Data-Age-Seconds` to the frontend, and the still-unconsumed `/series`/`/summary`/`/range` endpoints.

**L1 (15-min ingest) is deferred, not cancelled.** Measured projection with 15-min ingest is ~637 MiB at a 180-day raw window (~890 MiB if features densify too), which breaks the 512 MB M0 cap. Three ways forward when it is picked up: shorten the raw window to ~90 d (free, ~350 MiB, halves the training window), upgrade the Atlas tier (2 GB, paid), or keep hourly ingest. Do not start L1 without re-measuring storage first.

Gate before done: `code-reviewer` on the diff, `qa-engineer` when tests change, `database-engineer`/`sre-engineer` for migrations and SLOs. Never waive a blocker.

## User-side items (cannot be done from the repo)

- Sentry DSN on Render.
- Healthchecks.io checks for the ingest/inference/train crons (highest value: a dead-man's switch on `ingest.yml` + features; the in-repo `::warning` is ephemeral).
- Atlas project alerts (disk >80%, connections >80%, query targeting).
- Browser verification needs this session open in the desktop app, otherwise the browser tools report `disconnected`.
