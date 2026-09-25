# Handoff — operational notes for the next session

Read this **before** `docs/roadmap.md`. The roadmap holds the plan and the status
checklist; this file holds the environment, the hard rules and the traps that are not
written down anywhere else. Do not duplicate the roadmap here.

## Where things run

| Piece | Where | Notes |
| --- | --- | --- |
| Repo | `jomanor/weather-prediction-mlops`, branch `main` | Public repo → standard GitHub Actions minutes are unlimited. |
| API | Render service `meteoml-api`, deployed from `main` | `render.yaml`: docker, `backend/Dockerfile`, free plan, `autoDeploy: true`, health `GET /api/health`. Dashboard branch was the trap that caused stale deploys; it is now `main`. A deploy hook exists in the Render dashboard to force a deploy; the key is a secret, so it must never be committed to this public repo. |
| Frontend | Netlify site `weather-mlops-1787759329` | Linked to `main` with base dir `frontend`, `npm ci && npm run build`, publish `dist`. `netlify.toml` is the source of truth (base, NODE_VERSION 22, headers, SPA redirects, `/api/*` edge proxy to Render). Auto publishing must stay enabled. |
| DB | Atlas M0 `weather_db` (512 MB) | Measured 2026-09-25 `collStats.avgObjSize`: `raw_weather` 1362 B, `weather_features` 3856 B, `weather_predictions` 405 B. Steady-state projection **334 MiB** against a 384 MiB (75 %) budget and the 512 MiB cap → **177 MiB hard headroom**. `weather_features` is the largest collection (222 MiB, 43 % of the cap) and it is a single overwritten snapshot, not an accumulator. First breach axis: extending the raw window — `raw_weather` TTL > ~208 d breaks the 75 % budget, > ~281 d breaks 512 MiB. |
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

## State at handoff

- Batches 1–3 are merged and deployed (`main` = the Batch 3 squash commit). Batch 3
  delivered: the data-quality meter (`GET /api/weather/quality`), prediction intervals
  (residual-quantile p10/p90, `interval_level` 0.8) written to `weather_predictions` and
  shown as a band in the UI, honest metrics (temporal split by timestamp value,
  persistence/climatology baselines, skill score, Brier/prevalence/coverage, `n_test`,
  `split`, `interval`, `data_snapshot`, `commit` in `model_registry`), the frontend
  switch onto the bulk endpoints (clustered GeoJSON stations, weather-code icons, wind
  arrows, basemap switcher, ramp tokens, radar playback), and the R4 CI guardrails (real
  `mongo:6` index-backed-sort test, inference output-schema contract test, storage budget,
  `prune_models if: always()`, `scripts/pipeline_summary.py` step summaries).
- Verified live: `/api/weather/quality?days=7` returns 200 with per-city payloads; the
  newest trained temp model carries `interval`/`split`/`commit`/`data_snapshot`; a fresh
  inference run wrote 15 predictions with `temp_lower`/`temp_upper`/`interval_level=0.8`
  and honest GridFS lineage (`temp_prediction_1h_LinearRegression`).
- **Trap fixed this batch, do not regress it:** `inference.py` used to trust the MLflow
  registry for artifact metadata; when MLflow served the artifact the row lost its
  `interval` and reported fake `MLflow-latest` lineage. `_latest_model_entry` is now
  resolved first and always supplies metadata; MLflow only ever serves the *artifact*.
- **Live findings from the Batch 3 acceptance run (unresolved, Batch 4 candidates):**
  - All cities report `status:"bad"` because `age_hours` exceeds 6 h. `raw_weather` max was
    `17:30` while `weather_features` max was `14:00` (~3.5 h behind), at ~20:35 UTC. The
    declared hourly `ingest.yml` cron (`50 * * * *`) produced only four runs for the day
    (~6 h apart), so the meter's age thresholds (≤2 h ok / ≤6 h warn) are calibrated to a
    cadence the system does not currently achieve. Either fix the cadence (L1) or
    recalibrate the thresholds; until then the meter is noisy rather than actionable.
    Confirm the scheduler's behaviour before changing anything.
  - `weather_data` holds 0 rows — expected given the move to `raw_weather`, but the
    roadmap's R2 TTL entry for it is now vestigial.

## Next: Batch 4

Scope per `docs/roadmap.md` (L1 deferred by decision — see below): **M2** horizons
`[1,3,6,12,24]` as config, **M4** slice diagnostics + drift PSI, **D2** analytics
collection + endpoints, **U6** chart brushing/synced x-domain, command palette, keyboard
shortcuts, mobile affordances, **U7** fresh-data pulse, shimmer skeletons, 120–240 ms
transitions honouring `prefers-reduced-motion` (the weather-code icons part of U7 already
shipped in Batch 3). Also worth folding in: recalibrate the quality-meter age thresholds
against the real ingest cadence (see live findings above).

**L1 (15-min ingest) is deferred, not cancelled.** Measured projection with 15-min ingest
is ~637 MiB at a 180-day raw window (~890 MiB if features densify too), which breaks the
512 MB M0 cap. Three ways forward when it is picked up: shorten the raw window to ~90 d
(free, ~350 MiB, halves the training window), upgrade the Atlas tier (2 GB, paid), or
keep hourly ingest. Do not start L1 without re-measuring storage first.

Gate before done: `code-reviewer` on the diff, `qa-engineer` when tests change,
`database-engineer`/`sre-engineer` for migrations and SLOs. Never waive a blocker.

## User-side items (cannot be done from the repo)

Sentry DSN on Render; Healthchecks.io checks for the ingest/inference/train crons; Atlas
project alerts (disk >80%, connections >80%, query targeting). Browser verification needs
this session open in the desktop app, otherwise the browser tools report `disconnected`.
