import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useRadar, type RadarFrame, type RadarIndex } from '@/components/map/radar'

const FRAMES: RadarFrame[] = [
  { time: 1000, path: '/radar/p1', kind: 'past' },
  { time: 1200, path: '/radar/p2', kind: 'past' },
  { time: 1400, path: '/radar/n1', kind: 'nowcast' },
]

function loader(frames: RadarFrame[]): () => Promise<RadarIndex> {
  return vi.fn().mockResolvedValue({ host: 'https://tilecache.rainviewer.com', frames })
}

function setReducedMotion(reduce: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: query.includes('prefers-reduced-motion') ? reduce : false,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }))
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('useRadar playback', () => {
  it('opens on the most recent past frame, not the nowcast', async () => {
    setReducedMotion(false)
    const { result } = renderHook(() => useRadar(true, loader(FRAMES)))

    await waitFor(() => expect(result.current.status).toBe('ready'))
    expect(result.current.index).toBe(1)
    expect(result.current.current?.kind).toBe('past')
  })

  it('advances on the 700 ms loop and wraps to the start', async () => {
    setReducedMotion(false)
    const { result } = renderHook(() => useRadar(true, loader(FRAMES)))
    await waitFor(() => expect(result.current.status).toBe('ready'))

    vi.useFakeTimers()
    act(() => result.current.togglePlay())
    expect(result.current.playing).toBe(true)

    act(() => vi.advanceTimersByTime(700))
    expect(result.current.index).toBe(2)
    act(() => vi.advanceTimersByTime(700))
    expect(result.current.index).toBe(0)
  })

  it('blocks manual playback under prefers-reduced-motion but keeps the scrubber', async () => {
    setReducedMotion(true)
    const { result } = renderHook(() => useRadar(true, loader(FRAMES)))
    await waitFor(() => expect(result.current.status).toBe('ready'))

    expect(result.current.reducedMotion).toBe(true)
    vi.useFakeTimers()
    act(() => result.current.togglePlay())
    expect(result.current.playing).toBe(false)

    act(() => vi.advanceTimersByTime(2100))
    expect(result.current.index).toBe(1)

    // The scrubber itself still works.
    act(() => result.current.setIndex(2))
    expect(result.current.index).toBe(2)
  })

  it('never starts an interval with a single-frame timeline', async () => {
    setReducedMotion(false)
    const { result } = renderHook(() =>
      useRadar(true, loader([{ time: 1000, path: '/radar/only', kind: 'past' }])),
    )
    await waitFor(() => expect(result.current.status).toBe('ready'))

    vi.useFakeTimers()
    act(() => result.current.togglePlay())
    expect(result.current.playing).toBe(true)
    act(() => vi.advanceTimersByTime(2100))
    expect(result.current.index).toBe(0)
  })
})
