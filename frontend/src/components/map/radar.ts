import { useCallback, useEffect, useRef, useState } from 'react'

import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'

/**
 * RainViewer public radar (keyless).
 *
 * The index exposes a `past` loop and an optional `nowcast`. Frames are merged
 * into one ordered timeline; playback never autoplays, so
 * `prefers-reduced-motion` is honoured by construction (and explicitly stops an
 * active loop when the preference flips).
 */

export const RAINVIEWER_INDEX = 'https://api.rainviewer.com/public/weather-maps.json'

export interface RadarFrame {
  /** Frame time, epoch seconds. */
  time: number
  path: string
  kind: 'past' | 'nowcast'
}

export type RadarStatus = 'idle' | 'loading' | 'ready' | 'unavailable'

interface RainViewerIndex {
  host?: string
  radar?: {
    past?: Array<{ time?: number; path?: string }>
    nowcast?: Array<{ time?: number; path?: string }>
  }
}

/** Pure: merge `past` + `nowcast` into one chronological frame list. */
export function buildRadarFrames(index: unknown): RadarFrame[] {
  const parsed = (index ?? {}) as RainViewerIndex
  const frames: RadarFrame[] = []
  for (const [kind, list] of [
    ['past', parsed.radar?.past ?? []],
    ['nowcast', parsed.radar?.nowcast ?? []],
  ] as const) {
    for (const frame of list) {
      if (typeof frame.time === 'number' && Number.isFinite(frame.time) && frame.path) {
        frames.push({ time: frame.time, path: frame.path, kind })
      }
    }
  }
  return frames.sort((a, b) => a.time - b.time)
}

/** Tile template for a frame (`2` = universal blue, `1_1` = smooth, no snow). */
export function radarTileUrl(host: string, frame: RadarFrame): string {
  return `${host}${frame.path}/256/{z}/{x}/{y}/2/1_1.png`
}

let indexPromise: Promise<RadarIndex | null> | null = null

/** Fetches (and caches for the session) the keyless index. Never rejects. */
export function loadRadarIndex(): Promise<RadarIndex | null> {
  indexPromise ??= (async () => {
    try {
      const response = await fetch(RAINVIEWER_INDEX, { signal: AbortSignal.timeout(8000) })
      if (!response.ok) return null
      const payload: unknown = await response.json()
      const host = (payload as RainViewerIndex)?.host
      const frames = buildRadarFrames(payload)
      if (!host || frames.length === 0) return null
      return { host, frames }
    } catch {
      return null
    }
  })()
  return indexPromise
}

export interface RadarIndex {
  host: string
  frames: RadarFrame[]
}

export interface RadarPlayback {
  status: RadarStatus
  frames: RadarFrame[]
  host: string | null
  index: number
  current: RadarFrame | null
  playing: boolean
  /** True when the OS asks for reduced motion; playback is disabled. */
  reducedMotion: boolean
  setIndex: (index: number) => void
  togglePlay: () => void
}

/** Frame timeline + manual playback for the radar raster layer. */
export function useRadar(
  enabled: boolean,
  loadIndex: () => Promise<RadarIndex | null> = loadRadarIndex,
): RadarPlayback {
  const [status, setStatus] = useState<RadarStatus>('idle')
  const [host, setHost] = useState<string | null>(null)
  const [frames, setFrames] = useState<RadarFrame[]>([])
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const reducedMotion = usePrefersReducedMotion()
  /* The loader is kept in a ref so a caller passing an inline function cannot
     restart the fetch on every render. */
  const loadIndexRef = useRef(loadIndex)
  useEffect(() => {
    loadIndexRef.current = loadIndex
  }, [loadIndex])

  /* Fetch once per session, on first enable. `status` deliberately stays out
     of the dep list: flipping it to `loading` inside the effect would re-run
     and cancel the in-flight request. */
  const startedRef = useRef(false)
  const mountedRef = useRef(true)
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    if (!enabled || startedRef.current) return
    startedRef.current = true
    setStatus('loading')
    void loadIndexRef.current().then((loaded) => {
      if (!mountedRef.current) return
      if (!loaded) {
        setStatus('unavailable')
        return
      }
      setHost(loaded.host)
      setFrames(loaded.frames)
      /* Open on the most recent observed frame, not the nowcast. */
      const lastPast = loaded.frames.reduce((last, frame, i) => (frame.kind === 'past' ? i : last), 0)
      setIndex(lastPast)
      setStatus('ready')
    })
  }, [enabled])

  useEffect(() => {
    if (!enabled) setPlaying(false)
  }, [enabled])

  useEffect(() => {
    if (reducedMotion) setPlaying(false)
  }, [reducedMotion])

  /* Reduced motion blocks the loop entirely — not just autoplay — so a manual
     Play cannot start 700 ms frame stepping. The scrubber stays usable. */
  useEffect(() => {
    if (reducedMotion || !playing || frames.length < 2) return
    const timer = window.setInterval(() => {
      setIndex((current) => (current + 1) % frames.length)
    }, 700)
    return () => window.clearInterval(timer)
  }, [reducedMotion, playing, frames.length])

  const togglePlay = useCallback(() => {
    if (reducedMotion) return
    setPlaying((value) => !value)
  }, [reducedMotion])

  return {
    status,
    frames,
    host,
    index,
    current: frames[index] ?? null,
    playing,
    reducedMotion,
    setIndex,
    togglePlay,
  }
}
