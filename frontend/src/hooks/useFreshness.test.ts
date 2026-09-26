import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ageInSeconds, FRESH_WINDOW_SECONDS, useFreshness } from '@/hooks/useFreshness'

const NOW = new Date('2026-09-25T12:00:00Z').getTime()

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
    expect(result.current.isFresh).toBe(false)
  })

  it('tolerates one missed hourly inference cycle', () => {
    vi.setSystemTime(NOW)
    const { result } = renderHook(() => useFreshness('2026-09-25T10:30:00Z'))
    expect(result.current.ageSeconds).toBe(5400)
    expect(result.current.isFresh).toBe(true)
  })

  it('has no age for a missing timestamp', () => {
    const { result } = renderHook(() => useFreshness(null))
    expect(result.current.ageSeconds).toBeNull()
    expect(result.current.isFresh).toBe(false)
  })
})
