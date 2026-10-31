import { useCallback, useEffect, useState } from 'react'

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

/**
 * Server-measured age of the freshest observation, as reported by the API's
 * `X-Data-Age-Seconds` header. `receivedAt` is the local epoch ms when that
 * value was read, so the age can be carried forward without trusting the
 * client's wall clock for the observation timestamp.
 */
export interface ServerAge {
  ageSeconds: number
  receivedAt: number
}

/** Age of an ISO timestamp in whole seconds; null when absent/unparseable. */
export function ageInSeconds(iso: string | null | undefined, now = Date.now()): number | null {
  if (!iso) return null
  const parsed = new Date(iso).getTime()
  if (!Number.isFinite(parsed)) return null
  return Math.max(0, Math.round((now - parsed) / 1000))
}

/** Server age plus the seconds elapsed locally since the response landed. */
export function ageFromServer({ ageSeconds, receivedAt }: ServerAge, now = Date.now()): number {
  const elapsed = Math.max(0, Math.round((now - receivedAt) / 1000))
  return ageSeconds + elapsed
}

/** A usable server age: present, non-negative and an integer number of seconds. */
function usableServerAge(serverAge: ServerAge | null | undefined): ServerAge | null {
  if (!serverAge) return null
  const { ageSeconds, receivedAt } = serverAge
  if (!Number.isInteger(ageSeconds) || ageSeconds < 0) return null
  if (!Number.isFinite(receivedAt)) return null
  return serverAge
}

/**
 * The age to render: the highest value the available evidence supports.
 *
 * Invariant — `max(serverAge + elapsed, payloadAge)`. The header is *not* live
 * authority. `/weather/current` is served `public, max-age=30` with a strong
 * ETag, so a revalidation answers 304 and the browser replays the stored body
 * *and* the stored header while JavaScript sees an ordinary successful
 * response (a 304 is invisible to `fetch`). React Query then resets
 * `dataUpdatedAt`, so `serverAge + elapsed` restarts from a value measured at
 * the last real 200: with the pipeline stalled the payload stops changing,
 * every refetch is a 304, and that carried-forward figure freezes while the
 * data gets older. Taking the max means the header can only ever *raise* the
 * estimate, never lower it — a stale payload can never be painted fresh, which
 * is the one direction that must not fail. With no usable server age the
 * payload math is used unchanged; with no payload timestamp the carried
 * server age stands alone (the only evidence there is).
 */
export function effectiveAgeSeconds(
  iso: string | null | undefined,
  serverAge: ServerAge | null | undefined = null,
  now = Date.now(),
): number | null {
  const payloadAge = ageInSeconds(iso, now)
  const reported = usableServerAge(serverAge)
  if (!reported) return payloadAge
  const carried = ageFromServer(reported, now)
  return payloadAge === null ? carried : Math.max(carried, payloadAge)
}

/**
 * U7: derive the fresh-data pulse from the data's own age, not from every
 * render. The age is recomputed only when the timestamp changes and on a
 * one-minute tick, so a re-render cannot restart the pulse.
 *
 * `serverAge` only feeds a `max()` (see `effectiveAgeSeconds`), so a replayed
 * or frozen header cannot hide an ageing payload; a payload whose timestamp
 * disagrees with the header can also only make the data look staler, never
 * fresher. Absent both, there is no age to report.
 */
export function useFreshness(
  iso: string | null | undefined,
  freshWithinSeconds = FRESH_WINDOW_SECONDS,
  serverAge: ServerAge | null = null,
): Freshness {
  const reported = usableServerAge(serverAge)
  const reportedSeconds = reported?.ageSeconds ?? null
  const receivedAt = reported?.receivedAt ?? null

  const resolve = useCallback(
    (now: number) =>
      effectiveAgeSeconds(
        iso,
        reportedSeconds !== null && receivedAt !== null
          ? { ageSeconds: reportedSeconds, receivedAt }
          : null,
        now,
      ),
    [iso, reportedSeconds, receivedAt],
  )

  const [ageSeconds, setAgeSeconds] = useState<number | null>(() => resolve(Date.now()))

  useEffect(() => {
    setAgeSeconds(resolve(Date.now()))
    if (!iso && reportedSeconds === null) return
    const timer = window.setInterval(() => setAgeSeconds(resolve(Date.now())), TICK_MS)
    return () => window.clearInterval(timer)
  }, [iso, reportedSeconds, resolve])

  return {
    ageSeconds,
    isFresh: ageSeconds !== null && ageSeconds <= freshWithinSeconds,
  }
}
