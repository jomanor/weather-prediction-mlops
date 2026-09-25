import { useEffect, useState } from 'react'

/**
 * Default freshness window. Real cadence: inference runs hourly (around :05)
 * and features rebuild roughly every 6 h; L1's 15-min ingest is deferred.
 * 120 min tolerates one missed hourly cycle and still flags a stale feed.
 */
export const FRESH_WINDOW_SECONDS = 120 * 60
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
