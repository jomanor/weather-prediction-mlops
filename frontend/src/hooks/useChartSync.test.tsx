import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ChartSyncProvider, useChartSync } from '@/hooks/useChartSync'

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
  })
})
