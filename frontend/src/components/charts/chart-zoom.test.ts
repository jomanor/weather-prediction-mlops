import { describe, expect, it } from 'vitest'

import {
  MIN_ZOOM_SPAN_MS,
  canZoomIn,
  clampDomain,
  isZoomed,
  panDomain,
  resetDomain,
  unionDomain,
  zoomDomain,
} from '@/components/charts/chart-zoom'
import type { TimeDomain } from '@/hooks/useChartSync'

const HOUR = 3_600_000
/** Ten hours of reachable data. */
const BOUNDS: TimeDomain = [0, 10 * HOUR]

function span([start, end]: TimeDomain): number {
  return end - start
}

describe('clampDomain', () => {
  it('leaves a domain that already sits inside the bounds alone', () => {
    expect(clampDomain([2 * HOUR, 6 * HOUR], BOUNDS)).toEqual([2 * HOUR, 6 * HOUR])
  })

  it('slides a domain that overhangs the right edge back inside', () => {
    expect(clampDomain([6 * HOUR, 12 * HOUR], BOUNDS)).toEqual([4 * HOUR, 10 * HOUR])
  })

  it('slides a domain that overhangs the left edge forward', () => {
    expect(clampDomain([-2 * HOUR, 4 * HOUR], BOUNDS)).toEqual([0, 6 * HOUR])
  })

  it('caps a domain wider than the bounds', () => {
    expect(clampDomain([-5 * HOUR, 15 * HOUR], BOUNDS)).toEqual(BOUNDS)
  })

  it('grows a too-narrow domain to the minimum span around its centre', () => {
    const [start, end] = clampDomain([5 * HOUR, 5 * HOUR + 60_000], BOUNDS)
    expect(span([start, end])).toBe(MIN_ZOOM_SPAN_MS)
    expect((start + end) / 2).toBe(5 * HOUR + 30_000)
  })

  it('uses the whole bounds when they are shorter than the minimum span', () => {
    const short: TimeDomain = [HOUR, 2 * HOUR]
    expect(clampDomain([HOUR, HOUR + 60_000], short)).toEqual(short)
    expect(clampDomain([0, 5 * HOUR], short)).toEqual(short)
  })

  it('normalises an inverted domain', () => {
    expect(clampDomain([6 * HOUR, 2 * HOUR], BOUNDS)).toEqual([2 * HOUR, 6 * HOUR])
  })

  it('collapses to the bound edge when the bounds have no width', () => {
    expect(clampDomain([0, HOUR], [7, 7])).toEqual([7, 7])
  })
})

describe('zoomDomain', () => {
  it('shrinks the span around the anchor and keeps it in place', () => {
    const zoomed = zoomDomain([0, 10 * HOUR], 2, 5 * HOUR, BOUNDS)
    expect(zoomed).toEqual([2.5 * HOUR, 7.5 * HOUR])
    expect((5 * HOUR - zoomed[0]) / span(zoomed)).toBe(0.5)
  })

  it('keeps an off-centre anchor under the pointer', () => {
    const zoomed = zoomDomain(BOUNDS, 1.25, 2 * HOUR, BOUNDS)
    expect(zoomed[1] - zoomed[0]).toBeCloseTo((10 * HOUR) / 1.25, 6)
    expect((2 * HOUR - zoomed[0]) / span(zoomed)).toBeCloseTo(0.2, 6)
  })

  it('clamps an anchor outside the view to the nearest edge', () => {
    expect(zoomDomain(BOUNDS, 2, -5 * HOUR, BOUNDS)).toEqual([0, 5 * HOUR])
    expect(zoomDomain(BOUNDS, 2, 20 * HOUR, BOUNDS)).toEqual([5 * HOUR, 10 * HOUR])
  })

  it('stops at the minimum span', () => {
    const [start, end] = zoomDomain([4 * HOUR, 6 * HOUR], 10, 5 * HOUR, BOUNDS)
    expect(span([start, end])).toBe(MIN_ZOOM_SPAN_MS)
  })

  it('cannot zoom out past the bounds', () => {
    expect(zoomDomain(BOUNDS, 0.5, 5 * HOUR, BOUNDS)).toEqual(BOUNDS)
    expect(zoomDomain([2 * HOUR, 6 * HOUR], 0.5, 4 * HOUR, BOUNDS)).toEqual([
      0,
      8 * HOUR,
    ])
  })

  it('ignores a degenerate factor or width', () => {
    expect(zoomDomain(BOUNDS, 0, 5 * HOUR, BOUNDS)).toEqual(BOUNDS)
    expect(zoomDomain([5 * HOUR, 5 * HOUR], 2, 5 * HOUR, BOUNDS)).toEqual([
      4 * HOUR,
      6 * HOUR,
    ])
  })
})

describe('panDomain', () => {
  it('slides the window without changing its width', () => {
    const panned = panDomain([2 * HOUR, 6 * HOUR], HOUR, BOUNDS)
    expect(panned).toEqual([3 * HOUR, 7 * HOUR])
    expect(span(panned)).toBe(span([2 * HOUR, 6 * HOUR]))
  })

  it('clamps at both ends', () => {
    expect(panDomain([2 * HOUR, 6 * HOUR], -5 * HOUR, BOUNDS)).toEqual([0, 4 * HOUR])
    expect(panDomain([6 * HOUR, 9 * HOUR], 5 * HOUR, BOUNDS)).toEqual([7 * HOUR, 10 * HOUR])
  })

  it('cannot move a full-extent window', () => {
    expect(panDomain(BOUNDS, HOUR, BOUNDS)).toEqual(BOUNDS)
    expect(panDomain(BOUNDS, -HOUR, BOUNDS)).toEqual(BOUNDS)
  })

  it('is a no-op for a zero delta', () => {
    expect(panDomain([3 * HOUR, 7 * HOUR], 0, BOUNDS)).toEqual([3 * HOUR, 7 * HOUR])
  })
})

describe('resetDomain', () => {
  it('always returns the whole bounds', () => {
    expect(resetDomain(BOUNDS)).toEqual(BOUNDS)
  })

  it('handles degenerate bounds', () => {
    expect(resetDomain([3, 3])).toEqual([3, 3])
  })
})

describe('isZoomed', () => {
  it('is false while the view fills the bounds', () => {
    expect(isZoomed(BOUNDS, BOUNDS)).toBe(false)
    expect(isZoomed(null, BOUNDS)).toBe(false)
    expect(isZoomed(BOUNDS, null)).toBe(false)
  })

  it('is true once the view is narrower, ignoring sub-millisecond noise', () => {
    expect(isZoomed([HOUR, 9 * HOUR], BOUNDS)).toBe(true)
    expect(isZoomed([0, 10 * HOUR - 0.5], BOUNDS)).toBe(false)
  })
})

describe('canZoomIn', () => {
  it('allows zooming while the span is above the minimum', () => {
    expect(canZoomIn(BOUNDS, BOUNDS)).toBe(true)
    expect(canZoomIn([4 * HOUR, 6 * HOUR], BOUNDS)).toBe(false)
    expect(canZoomIn(null, BOUNDS)).toBe(false)
  })

  it('respects bounds shorter than the minimum span', () => {
    const short: TimeDomain = [0, HOUR]
    expect(canZoomIn(short, short)).toBe(false)
  })
})

describe('unionDomain', () => {
  it('widens both edges and passes through a missing side', () => {
    expect(unionDomain([0, 4], [2, 9])).toEqual([0, 9])
    expect(unionDomain(null, [2, 9])).toEqual([2, 9])
    expect(unionDomain([2, 9], null)).toEqual([2, 9])
    expect(unionDomain(null, null)).toBeNull()
  })
})
