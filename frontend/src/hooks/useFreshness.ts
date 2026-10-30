import { useEffect, useState } from 'react'

/**
 * Default freshness window, aligned with the cadence the pipeline actually
 * achieves: GitHub Actions throttles the declared hourly crons to ~4–7 runs a
 * day, so observations arrive 6–7 h apart (see docs/roadmap.md). A 12 h window
 * matches the quality meter's `ok` age gate — the earlier 120 min assumed an
 * hourly ingest that does not exist and painted healthy data as stale.
 */
export const FRESH_WINDOW_SECONDS = 12 * 60 * 60
const TICK_MS = 60_000

export interface Freshness {
  ageSeconds: number | null
  /** True while the observation is inside the freshness window. */
  isFresh: boolean
}

/** Age of an ISO timestamp in whole seconds; null when absent/unparseable. */
export function ageInSeconds(iso: string | null | undefined, now = Date.now()): number | null {
  if (!iso) return null
  const parsed = new Date(iso).getTime()
  if (!Number.isFinite(parsed)) return null
  return Math.max(0, Math.round((now - parsed) / 1000))
}

/**
 * U7: derive the fresh-data pulse from the data's own age, not from every
 * render. The age is recomputed only when the timestamp changes and on a
 * one-minute tick, so a re-render cannot restart the pulse.
 */
export function useFreshness(
  iso: string | null | undefined,
  freshWithinSeconds = FRESH_WINDOW_SECONDS,
): Freshness {
  const [ageSeconds, setAgeSeconds] = useState<number | null>(() => ageInSeconds(iso))

  useEffect(() => {
    setAgeSeconds(ageInSeconds(iso))
    if (!iso) return
    const timer = window.setInterval(() => setAgeSeconds(ageInSeconds(iso)), TICK_MS)
    return () => window.clearInterval(timer)
  }, [iso])

  return {
    ageSeconds,
    isFresh: ageSeconds !== null && ageSeconds <= freshWithinSeconds,
  }
}
