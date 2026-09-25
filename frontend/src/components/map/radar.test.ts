import { describe, expect, it } from 'vitest'

import { buildRadarFrames, radarTileUrl } from '@/components/map/radar'

const INDEX = {
  host: 'https://tilecache.rainviewer.com',
  radar: {
    past: [
      { time: 1000, path: '/v2/radar/past1' },
      { time: 1200, path: '/v2/radar/past2' },
    ],
    nowcast: [{ time: 1400, path: '/v2/radar/future1' }],
  },
}

describe('buildRadarFrames', () => {
  it('merges past and nowcast into one chronological timeline', () => {
    const frames = buildRadarFrames(INDEX)
    expect(frames.map((frame) => frame.time)).toEqual([1000, 1200, 1400])
    expect(frames.map((frame) => frame.kind)).toEqual(['past', 'past', 'nowcast'])
  })

  it('keeps working without a nowcast block', () => {
    expect(buildRadarFrames({ radar: { past: INDEX.radar.past } })).toHaveLength(2)
  })

  it('drops malformed frames and handles a missing index', () => {
    const frames = buildRadarFrames({
      radar: { past: [{ time: 1 }, { path: '/x' }, { time: Number.NaN, path: '/y' }] },
    })
    expect(frames).toEqual([])
    expect(buildRadarFrames(null)).toEqual([])
    expect(buildRadarFrames(undefined)).toEqual([])
  })
})

describe('radarTileUrl', () => {
  it('builds the RainViewer tile template for a frame', () => {
    expect(radarTileUrl('https://tilecache.rainviewer.com', { time: 1, path: '/v2/radar/x', kind: 'past' })).toBe(
      'https://tilecache.rainviewer.com/v2/radar/x/256/{z}/{x}/{y}/2/1_1.png',
    )
  })
})
