import { describe, expect, it } from 'vitest'

import { brushDomain } from '@/components/charts/chart-brush'

const ROWS = [{ ts: 100 }, { ts: 200 }, { ts: 300 }, { ts: 400 }]

describe('brushDomain', () => {
  it('maps an index range onto its timestamps', () => {
    expect(brushDomain(ROWS, { startIndex: 1, endIndex: 3 })).toEqual([200, 400])
  })

  it('clamps out-of-range indices', () => {
    expect(brushDomain(ROWS, { startIndex: -5, endIndex: 99 })).toEqual([100, 400])
  })

  it('defaults to the full extent when the range omits an index', () => {
    expect(brushDomain(ROWS, {})).toEqual([100, 400])
    expect(brushDomain(ROWS, { startIndex: 2 })).toEqual([300, 400])
  })

  it('rejects degenerate selections', () => {
    expect(brushDomain(ROWS, { startIndex: 2, endIndex: 2 })).toBeNull()
    expect(brushDomain(ROWS, { startIndex: 3, endIndex: 1 })).toBeNull()
    expect(brushDomain([], { startIndex: 0, endIndex: 0 })).toBeNull()
    expect(brushDomain([{ ts: 1 }], {})).toBeNull()
  })
})
