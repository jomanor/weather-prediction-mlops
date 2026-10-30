import type { Prediction } from '@/api/schemas'

export const HOUR_MS = 3_600_000

/** Epoch ms of a prediction's source observation, or null when unparseable. */
function sourceMs(prediction: Prediction): number | null {
  const ts = new Date(prediction.source_timestamp).getTime()
  return Number.isFinite(ts) ? ts : null
}

/**
 * The newest inference run: the rows sharing the maximum `source_timestamp`,
 * one per horizon.
 *
 * `GET /api/predictions/{city}` returns *history*, chronological oldest →
 * newest, so `predictions[0]` is the oldest run in the window — not the
 * latest forecast. Batch jobs may also emit more than one row for the same
 * `(source_timestamp, horizon_hours)`; the first one seen wins.
 */
export function latestRun(predictions: Prediction[]): Prediction[] {
  let newest: number | null = null
  for (const prediction of predictions) {
    const ts = sourceMs(prediction)
    if (ts !== null && (newest === null || ts > newest)) newest = ts
  }
  if (newest === null) return []

  const seenHorizons = new Set<number>()
  const run: Prediction[] = []
  for (const prediction of predictions) {
    if (sourceMs(prediction) !== newest) continue
    if (seenHorizons.has(prediction.horizon_hours)) continue
    seenHorizons.add(prediction.horizon_hours)
    run.push(prediction)
  }
  return run.sort((a, b) => a.horizon_hours - b.horizon_hours)
}

/** Target instant a prediction describes: source observation + horizon. */
export function forecastTargetMs(prediction: Prediction): number | null {
  const source = sourceMs(prediction)
  if (source === null) return null
  const horizon = Number.isFinite(prediction.horizon_hours) ? prediction.horizon_hours : 0
  return source + horizon * HOUR_MS
}

/**
 * Card row for the selected horizon from an already-derived latest run.
 *
 * When the run has no row for that horizon (the backend may not have written
 * it yet), fall back to the newest forecast by `prediction_timestamp` so the
 * card never blanks out or crashes over a missing horizon.
 */
export function pickHorizon(run: Prediction[], horizon: number): Prediction | undefined {
  const exact = run.find((prediction) => prediction.horizon_hours === horizon)
  if (exact) return exact

  let fallback: Prediction | undefined
  let fallbackMs = -Infinity
  for (const prediction of run) {
    const ts = new Date(prediction.prediction_timestamp).getTime()
    if (Number.isFinite(ts) && ts > fallbackMs) {
      fallback = prediction
      fallbackMs = ts
    }
  }
  return fallback ?? run[0]
}
