# Handoff — operational notes for the next session

Read this **before** `docs/roadmap.md`. The roadmap holds the plan and the status
checklist; this file holds the environment, the hard rules and the traps that are not
written down anywhere else. Do not duplicate the roadmap here.

## Where things run

| Piece | Where | Notes |
| --- | --- | --- |
| Repo | `jomanor/weather-prediction-mlops`, branch `main` | Public repo → standard GitHub Actions minutes are unlimited. |
| API | Render service `meteoml-api`, deployed from `main` | `render.yaml`: docker, `backend/Dockerfile`, free plan, `autoDeploy: true`, health `GET /api/health`. Dashboard branch was the trap that caused stale deploys; it is now `main`. |
| Frontend | Netlify site `weather-mlops-1787759329` | Linked to `main` with base dir `frontend`, `npm ci && npm run build`, publish `dist`. `netlify.toml` is the source of truth (base, NODE_VERSION 22, headers, SPA redirects, `/api/*` edge proxy to Render). Auto publishing must stay enabled. |
| DB | Atlas M0 `weather_db` (512 MB) | Growing ~1.4 MB/day from `weather_features`. TTLs are installed. |
| Cron | GitHub Actions | `ingest.yml` hourly `:50`, `ml-pipeline.yml` inference `:05`, features `20 */6`, train `35 2`, plus `backfill.yml` (manual). |

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
   carries the feature list it was trained with.
2. **Every Mongo sort must be prefixed by `city`** (`{city:1, timestamp:-1}`), and
   aggregates pass `allowDiskUse=True`. The `tag: 292` failures were an unindexed
   blocking sort. Never add a global `{timestamp:-1}` index. Do not rely on
   `allowDiskUse` alone — Atlas M0 rejects external sorting.
3. **Do not rename collections**, do not split `weather_data`/`raw_weather`, and do not
   change the prediction key `(city, source_timestamp, horizon_hours)` — model lineage and
   the backfill depend on them.
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

Render: hit `GET /api/health/ready` (DB ping) — a 404 means the new code is not deployed.

To check that live payloads still satisfy the frontend contract, drop a temporary
`frontend/src/api/live-contract.test.ts` that fetches each endpoint and `safeParse`s it
with the schemas from `src/api/schemas.ts`, run it with `./node_modules/.bin/vitest run
<file>`, then delete the file. That is how a whole class of "the app is broken" reports
was ruled out.

## State at handoff

- Batch 1 (correctness) and Batch 2 (cache/ETag, bulk reads, feature registry, lazy
  routes, URL state, cross-filtering) are merged and deployed; the API and the Netlify
  bundle both match `main`.
- Atlas: index + TTL set installed; `model_registry` rows carry `model_type`, `stage`,
  `schema_version: 2` and real metrics for the newest models; `weather_predictions`
  writes with a stable string `_id` (upsert).
- The frontend does **not** consume any Batch 2 endpoint yet (`/weather/series`,
  `/weather/summary`, `/weather/range`, `/map/stations`) — Batch 3 should switch the
  overview, map and charts onto the bulk endpoints.
- `docs/roadmap.md` → `## Accepted follow-ups` lists the deferred items; keep it current.

## Next: Batch 3

Scope per `docs/roadmap.md`: **U3** radar playback, **U4** data colour ramps
(`--temp-1..7`, `--rain-1..5`, `--wind-1..4` in `frontend/src/styles/index.css`),
**U5** clustered GeoJSON markers + wind arrows + weather-code icons + basemap switcher,
**D1** data-quality meter, **M1** honest metrics (temporal split, persistence baseline,
skill score), **D3** prediction intervals. `frontend.config.json` pins the design tokens
and `MAP_PAINT` in `lib/chart-theme.ts` holds the MapLibre literals.

Gate before done: `code-reviewer` on the diff, `qa-engineer` when tests change,
`database-engineer`/`sre-engineer` for migrations and SLOs. Never waive a blocker.

## User-side items (cannot be done from the repo)

Sentry DSN on Render; Healthchecks.io checks for the ingest/inference/train crons; Atlas
project alerts (disk >80%, connections >80%, query targeting). Browser verification needs
this session open in the desktop app, otherwise the browser tools report `disconnected`.
