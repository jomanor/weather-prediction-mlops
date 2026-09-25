import { act, render, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import {
  ChartSyncProvider,
  useChartSync,
  useChartWindow,
  type TimeDomain,
} from '@/hooks/useChartSync'

const ANCHOR = 1_000_000_000_000
const HOUR_MS = 3_600_000

function useProbe() {
  return useChartSync()
}

describe('useChartSync', () => {
  it('stays detached (null domain) without a provider', () => {
    const { result } = renderHook(() => useProbe())
    expect(result.current.domain).toBeNull()
    act(() => result.current.setWindow(24, ANCHOR))
    expect(result.current.domain).toBeNull()
  })

  it('anchors a window of the requested length at the anchor', () => {
    const { result } = renderHook(() => useProbe(), { wrapper: ChartSyncProvider })
    act(() => result.current.setWindow(24, ANCHOR))
    expect(result.current.domain).toEqual([ANCHOR - 24 * HOUR_MS, ANCHOR])
    expect(result.current.brushed).toBe(false)
  })

  it('publishes a brushed domain and clears it on reset', () => {
    const { result } = renderHook(() => useProbe(), { wrapper: ChartSyncProvider })
    act(() => result.current.setDomain([ANCHOR, ANCHOR + HOUR_MS]))
    expect(result.current.domain).toEqual([ANCHOR, ANCHOR + HOUR_MS])
    expect(result.current.brushed).toBe(true)

    act(() => result.current.reset())
    expect(result.current.domain).toBeNull()
    expect(result.current.brushed).toBe(false)
  })

  it('rejects an empty or inverted brush selection', () => {
    const { result } = renderHook(() => useProbe(), { wrapper: ChartSyncProvider })
    act(() => result.current.setDomain([ANCHOR, ANCHOR]))
    expect(result.current.domain).toBeNull()
  })
})

describe('useChartWindow leak fix', () => {
  it('resets the shared domain when the owning page unmounts', () => {
    let seen: TimeDomain | null = null
    function Observer() {
      seen = useChartSync().domain
      return null
    }
    function Page() {
      useChartWindow(24, new Date(ANCHOR).toISOString())
      return null
    }

    const { rerender } = render(
      <ChartSyncProvider>
        <Observer />
        <Page />
      </ChartSyncProvider>,
    )
    expect(seen).toEqual([ANCHOR - 24 * HOUR_MS, ANCHOR])

    rerender(
      <ChartSyncProvider>
        <Observer />
      </ChartSyncProvider>,
    )
    expect(seen).toBeNull()
  })
})
