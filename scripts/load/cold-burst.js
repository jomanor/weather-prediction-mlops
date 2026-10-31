/**
 * Cold-burst load check (roadmap: "No k6 cold-burst test yet").
 *
 * Per iteration a VU fires, CONCURRENTLY (``http.batch``), the three cached
 * analytics views the dashboard opens with plus ``/api/v1/health``:
 *
 *   GET /api/v1/analytics/daily?city=<CITY>&days=<DAYS>   (city is required)
 *   GET /api/v1/analytics/diurnal?days=<DAYS>
 *   GET /api/v1/analytics/correlation?days=<DAYS>
 *   GET /api/v1/health
 *
 * The point is the COLD path, against a free Render instance that sleeps after
 * ~15 min idle: the first request of the run can take 30-60 s to boot uvicorn
 * plus open Mongo. So:
 *
 *   * every request gets a 60 s timeout (no retries -- a cold failure must be
 *     recorded, never papered over);
 *   * the cold burst (VU 1 / iteration 0) is ALSO recorded on its own Trend,
 *     ``cold_start_ms``: the slowest of those four responses. Without it the
 *     boot latency would only be visible as one anonymous sample inside the
 *     overall p95. Note that those samples are still counted in
 *     ``analytics_ms``/``http_req_failed`` -- the budget is the budget; the
 *     cold trend is for diagnosis, not an exemption.
 *   * the cold burst is GATED, not just reported: ``cold_start_ok`` must be
 *     > 0, i.e. all four probe responses were 200. Four 60 s timeouts on a
 *     sleeping instance are 4 requests out of ~2400 in a 45 s run (~0.2 %):
 *     they pass ``health_failed`` (< 1 %), ``http_req_failed`` (< 5 %) and the
 *     analytics p95 alike, so without this threshold the exact failure the
 *     script exists to catch would exit 0.
 *   * ``gracefulStop`` is 120 s so an in-flight 60 s cold request is not
 *     truncated by the end of a 45 s duration.
 *
 * Load defaults are deliberately modest (3 VUs / 45 s, no target rate) because
 * the production target is a single-core free instance. Override with env vars:
 *
 *   BASE_URL=https://host VUS=3 DURATION=45s CITY=Madrid DAYS=90 \
 *     k6 run scripts/load/cold-burst.js --summary-export=/tmp/summary.json
 */

import http from 'k6/http';
import { check } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = (__ENV.BASE_URL || 'https://weather-prediction-mlops.onrender.com').replace(/\/+$/, '');
const VUS = Number(__ENV.VUS || 3);
const DURATION = __ENV.DURATION || '45s';
const CITY = __ENV.CITY || 'Madrid';
const DAYS = Number(__ENV.DAYS || 90);
const TIMEOUT = '60s';

// Custom metrics: analytics_ms (the group p95 budget), health_failed (the
// health-check failure rate), and the cold probe (cold_start_ms to read,
// cold_start_ok to gate on).
const analyticsMs = new Trend('analytics_ms', true);
const healthFailed = new Rate('health_failed');
const coldStartMs = new Trend('cold_start_ms', true);
const coldStartOk = new Rate('cold_start_ok');

export const options = {
  scenarios: {
    cold_burst: {
      executor: 'constant-vus',
      vus: VUS,
      duration: DURATION,
      // An iteration that started cold may need the full 60 s timeout to finish.
      gracefulStop: '120s',
    },
  },
  discardResponseBodies: true,
  thresholds: {
    // Analytics group: p95 under 2 s. Warm cache hits are ~10-50 ms; anything
    // near the budget is a cold recompute or a saturated single core.
    analytics_ms: ['p(95)<2000'],
    // Health-check failures (non-200, network error or 60 s timeout) < 1 %.
    health_failed: ['rate<0.01'],
    // Overall failure budget, informational ceiling for the other endpoints.
    http_req_failed: ['rate<0.05'],
    // The cold probe must have got all four 200s (one sample per run: 1 = pass,
    // 0 = fail). Separate from the rates above because ~0.2 % of a run's
    // requests can be silent inside them. Not an exemption: the cold samples
    // still count in analytics_ms and http_req_failed.
    cold_start_ok: ['rate>0'],
  },
};

function get(url, surface) {
  return {
    method: 'GET',
    url: url,
    params: { timeout: TIMEOUT, tags: { surface: surface } },
  };
}

export default function () {
  // The designated cold probe: the very first iteration of the very first VU.
  const coldProbe = __VU === 1 && __ITER === 0;

  const res = http.batch({
    daily: get(`${BASE_URL}/api/v1/analytics/daily?city=${encodeURIComponent(CITY)}&days=${DAYS}`, 'analytics'),
    diurnal: get(`${BASE_URL}/api/v1/analytics/diurnal?days=${DAYS}`, 'analytics'),
    correlation: get(`${BASE_URL}/api/v1/analytics/correlation?days=${DAYS}`, 'analytics'),
    health: get(`${BASE_URL}/api/v1/health`, 'health'),
  });

  let worstColdMs = 0;
  let coldAllOk = true;

  Object.keys(res).forEach((name) => {
    const r = res[name];
    const ok = r.status === 200;
    check(r, { [`${name} 200`]: (x) => x.status === 200 });

    if (name === 'health') {
      healthFailed.add(!ok);
    } else {
      analyticsMs.add(r.timings.duration);
    }

    if (coldProbe) {
      worstColdMs = Math.max(worstColdMs, r.timings.duration || 0);
      coldAllOk = coldAllOk && ok;
    }
  });

  if (coldProbe) {
    coldStartMs.add(worstColdMs);
    // One sample per run; thresholds cold_start_ok: rate>0 makes a single
    // not-all-200 probe fail the run instead of only colouring the summary.
    coldStartOk.add(coldAllOk);
  }
}
