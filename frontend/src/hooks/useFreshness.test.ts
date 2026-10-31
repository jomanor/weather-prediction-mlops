import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ageFromServer,
  ageInSeconds,
  effectiveAgeSeconds,
  FRESH_WINDOW_SECONDS,
  useFreshness,
  type ServerAge,
} from '@/hooks/useFreshness'

const NOW = new Date('2026-09-25T12:00:00Z').getTime()

/** A response that landed `secondsAgo` before NOW reporting `ageSeconds`. */
function serverAge(ageSeconds: number, secondsAgo = 0): ServerAge {
  return { ageSeconds, receivedAt: NOW - secondsAgo * 1000 }
}

describe('ageInSeconds', () => {
  it('returns null for absent or invalid timestamps', () => {
    expect(ageInSeconds(null, NOW)).toBeNull()
    expect(ageInSeconds(undefined, NOW)).toBeNull()
    expect(ageInSeconds('not-a-date', NOW)).toBeNull()
  })

  it('rounds to whole seconds and never goes negative', () => {
    expect(ageInSeconds('2026-09-25T11:59:30Z', NOW)).toBe(30)
    expect(ageInSeconds('2026-09-25T12:05:00Z', NOW)).toBe(0)
  })
})

describe('ageFromServer', () => {
  it('adds the local elapsed time to the server age', () => {
    expect(ageFromServer(serverAge(600), NOW)).toBe(600)
    expect(ageFromServer(serverAge(600, 30), NOW)).toBe(630)
  })

  it('never goes backwards when the clock baseline is ahead', () => {
    expect(ageFromServer({ ageSeconds: 10, receivedAt: NOW + 5_000 }, NOW)).toBe(10)
  })
})

/**
 * F1: `X-Data-Age-Seconds` is a floor, not live authority. `/weather/current`
 * is cacheable, so a 304 revalidation replays the header stored at the last 200
 * while the browser resets the local baseline — the carried-forward value then
 * freezes. `max()` is what stops a stalled pipeline from looking fresh.
 */
describe('effectiveAgeSeconds (server age can only raise the estimate)', () => {
  it('keeps the payload age when the replayed header is older (304 stall)', () => {
    /* Stalled pipeline: payload unchanged for 20 h, every refetch a 304, header
       replayed at 300 s with a fresh baseline. Payload evidence wins. */
    expect(effectiveAgeSeconds('2026-09-24T16:00:00Z', serverAge(300), NOW)).toBe(20 * 3600)
  })

  it('keeps the server age when it is the larger of the two', () => {
    /* Payload timestamp looks 5 min old (skewed client clock); the API measured
       13 h. The header raises the estimate past the window. */
    expect(effectiveAgeSeconds('2026-09-25T11:55:00Z', serverAge(13 * 3600), NOW)).toBe(
      13 * 3600,
    )
  })

  it('carries the server age forward when the payload timestamp agrees', () => {
    /* Same observation seen from both sides: the larger (older) reading wins. */
    expect(effectiveAgeSeconds('2026-09-25T11:50:00Z', serverAge(600), NOW)).toBe(600)
    expect(effectiveAgeSeconds('2026-09-25T11:49:00Z', serverAge(600), NOW)).toBe(660)
  })

  it('falls back to the payload timestamp when there is no header', () => {
    expect(effectiveAgeSeconds('2026-09-25T11:00:00Z', null, NOW)).toBe(3600)
    expect(effectiveAgeSeconds('2026-09-25T11:00:00Z', undefined, NOW)).toBe(3600)
  })

  it('uses the server age alone when the payload has no timestamp', () => {
    expect(effectiveAgeSeconds(null, serverAge(7200), NOW)).toBe(7200)
    expect(effectiveAgeSeconds('sin-dato', serverAge(7200, 60), NOW)).toBe(7260)
  })

  it('has no age when neither source is usable', () => {
    expect(effectiveAgeSeconds(null, null, NOW)).toBeNull()
    expect(effectiveAgeSeconds(undefined, serverAge(-5), NOW)).toBeNull()
  })
})

describe('useFreshness', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('is fresh inside the window and ages out on the tick', () => {
    vi.setSystemTime(NOW)
    const { result } = renderHook(() => useFreshness('2026-09-25T11:59:00Z'))
    expect(result.current.isFresh).toBe(true)
    expect(result.current.ageSeconds).toBe(60)

    act(() => vi.advanceTimersByTime(FRESH_WINDOW_SECONDS * 1000))
    expect(result.current.isFresh).toBe(false)
  })

  it('recomputes when the data timestamp changes, not on every render', () => {
    vi.setSystemTime(NOW)
    const { result, rerender } = renderHook(
      ({ iso }: { iso: string }) => useFreshness(iso),
      { initialProps: { iso: '2026-09-25T11:59:00Z' } },
    )
    expect(result.current.ageSeconds).toBe(60)

    rerender({ iso: '2026-09-25T09:00:00Z' })
    expect(result.current.ageSeconds).toBe(10800)
    expect(result.current.isFresh).toBe(true)
  })

  it('tolerates one missed hourly inference cycle', () => {
    vi.setSystemTime(NOW)
    const { result } = renderHook(() => useFreshness('2026-09-25T10:30:00Z'))
    expect(result.current.ageSeconds).toBe(5400)
    expect(result.current.isFresh).toBe(true)
  })

  it('keeps a 6 h observation fresh, matching the real ingest cadence', () => {
    vi.setSystemTime(NOW)
    const { result } = renderHook(() => useFreshness('2026-09-25T06:00:00Z'))
    expect(result.current.ageSeconds).toBe(6 * 3600)
    expect(result.current.isFresh).toBe(true)
  })

  it('has no age for a missing timestamp', () => {
    const { result } = renderHook(() => useFreshness(null))
    expect(result.current.ageSeconds).toBeNull()
    expect(result.current.isFresh).toBe(false)
  })

  it('lets a disagreeing payload timestamp only make the data look staler', () => {
    vi.setSystemTime(NOW)
    /* Client clock says 20 h old; the API measured 5 min. The header can no
       longer overrule the payload downwards — that is the direction that let a
       304-replayed header paint stale data fresh — so the max wins. */
    const { result } = renderHook(() =>
      useFreshness('2026-09-24T16:00:00Z', FRESH_WINDOW_SECONDS, serverAge(300)),
    )
    expect(result.current.ageSeconds).toBe(20 * 3600)
    expect(result.current.isFresh).toBe(false)
  })

  it('survives a 304 replay storm without freezing the age', () => {
    vi.setSystemTime(NOW)
    /* Frozen payload, header replayed at 300 s on every revalidation: the
       rendered age must follow the payload, not the replayed header. */
    const { result, rerender } = renderHook(
      ({ receivedAt }: { receivedAt: number }) =>
        useFreshness('2026-09-24T16:00:00Z', FRESH_WINDOW_SECONDS, {
          ageSeconds: 300,
          receivedAt,
        }),
      { initialProps: { receivedAt: NOW } },
    )
    expect(result.current.ageSeconds).toBe(20 * 3600)

    /* Each 304 resets the baseline while the payload stays put. */
    act(() => vi.advanceTimersByTime(5 * 60_000))
    rerender({ receivedAt: NOW + 5 * 60_000 })
    expect(result.current.ageSeconds).toBe(20 * 3600 + 300)
    expect(result.current.isFresh).toBe(false)
  })

  it('trusts the server age even when the payload timestamp is missing', () => {
    vi.setSystemTime(NOW)
    const { result } = renderHook(() =>
      useFreshness(null, FRESH_WINDOW_SECONDS, serverAge(7200)),
    )
    expect(result.current.ageSeconds).toBe(7200)
    expect(result.current.isFresh).toBe(true)
  })

  it('ages the server value forward across the tick instead of freezing it', () => {
    vi.setSystemTime(NOW)
    const { result } = renderHook(() =>
      useFreshness('2026-09-25T11:59:00Z', FRESH_WINDOW_SECONDS, serverAge(600)),
    )
    expect(result.current.ageSeconds).toBe(600)

    act(() => vi.advanceTimersByTime(2 * 60_000))
    expect(result.current.ageSeconds).toBe(720)

    /* The window is measured against the carried-forward value. */
    act(() => vi.advanceTimersByTime((FRESH_WINDOW_SECONDS - 720) * 1000))
    expect(result.current.isFresh).toBe(true)
    act(() => vi.advanceTimersByTime(2 * 60_000))
    expect(result.current.isFresh).toBe(false)
  })

  it('falls back to the payload timestamp when the header is absent', () => {
    vi.setSystemTime(NOW)
    const { result } = renderHook(() =>
      useFreshness('2026-09-25T11:00:00Z', FRESH_WINDOW_SECONDS, null),
    )
    expect(result.current.ageSeconds).toBe(3600)
    expect(result.current.isFresh).toBe(true)
  })

  it('falls back when the server age is negative or fractional', () => {
    vi.setSystemTime(NOW)
    const negative = renderHook(() =>
      useFreshness('2026-09-25T11:00:00Z', FRESH_WINDOW_SECONDS, serverAge(-30)),
    )
    expect(negative.result.current.ageSeconds).toBe(3600)
    negative.unmount()

    const fractional = renderHook(() =>
      useFreshness('2026-09-25T11:00:00Z', FRESH_WINDOW_SECONDS, serverAge(1.5)),
    )
    expect(fractional.result.current.ageSeconds).toBe(3600)
  })
})
