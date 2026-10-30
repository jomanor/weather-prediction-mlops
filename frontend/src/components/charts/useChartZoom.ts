import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'

import { clamp } from '@/lib/format'
import type { TimeDomain } from '@/hooks/useChartSync'
import {
  ZOOM_FACTOR,
  canZoomIn,
  clampDomain,
  isZoomed,
  panDomain,
  zoomDomain,
} from '@/components/charts/chart-zoom'

/** Horizontal insets between the chart box and its plotting area. */
export interface PlotInsets {
  left: number
  right: number
}

interface UseChartZoomOptions {
  /** The domain currently drawn on the x axis (`null` before there is data). */
  view: TimeDomain | null
  /** Zoom-out limit: the page window or the data extent. */
  bounds: TimeDomain | null
  /** Publishes the new view, or `null` to drop the zoom entirely. */
  onChange: (domain: TimeDomain | null) => void
  insets: PlotInsets
  /** Span divisor per wheel notch / button press. */
  factor?: number
}

export interface ChartZoomHandlers {
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerUp: (event: ReactPointerEvent<HTMLDivElement>) => void
  onPointerCancel: (event: ReactPointerEvent<HTMLDivElement>) => void
  onDoubleClick: () => void
}

export interface ChartZoom {
  /** Attach to the element that should sense the wheel/pointer/double click. */
  surfaceRef: (node: HTMLDivElement | null) => void
  zoomed: boolean
  canZoomIn: boolean
  zoomIn: () => void
  zoomOut: () => void
  reset: () => void
  handlers: ChartZoomHandlers
}

/** A pointer has to travel this far before a press becomes a pan. */
const DRAG_THRESHOLD_PX = 3

/**
 * Wheel-to-zoom, drag-to-pan and double-click-to-reset for a time axis.
 *
 * The chart keeps ownership of the domain (it renders from `view` and publishes
 * through `onChange`), so the same hook can drive the shared chart domain or a
 * local one. Geometry is read from the wrapper element with the axis gutters
 * subtracted, so the zoom anchors at the pointer rather than at the chart edge.
 *
 * The wheel listener is attached by hand because React registers `wheel` as a
 * passive root listener: `preventDefault` from `onWheel` cannot stop the page
 * from scrolling, and a chart that zooms while the page moves is unusable.
 */
export function useChartZoom({
  view,
  bounds,
  onChange,
  insets,
  factor = ZOOM_FACTOR,
}: UseChartZoomOptions): ChartZoom {
  const [surface, setSurface] = useState<HTMLDivElement | null>(null)
  /* Handlers outlive a render (listeners, pointer captures), so they read the
     latest props from a ref instead of closing over a stale domain. */
  const latest = useRef({ view, bounds, onChange })
  useEffect(() => {
    latest.current = { view, bounds, onChange }
  })

  const surfaceRef = useCallback((node: HTMLDivElement | null) => setSurface(node), [])

  const plotWidth = (element: HTMLElement) =>
    Math.max(0, element.getBoundingClientRect().width - insets.left - insets.right)

  /* Every interaction funnels through here: clamp to the bounds once, then
     report "back to full extent" as `null` so callers need no epsilon check. */
  const apply = useCallback((next: TimeDomain) => {
    const { bounds, onChange } = latest.current
    if (!bounds) return
    const clamped = clampDomain(next, bounds)
    onChange(isZoomed(clamped, bounds) ? clamped : null)
  }, [])

  const zoomBy = useCallback(
    (step: number) => {
      const { view, bounds } = latest.current
      if (!view || !bounds) return
      const anchor = (view[0] + view[1]) / 2
      apply(zoomDomain(view, step, anchor, bounds))
    },
    [apply],
  )

  const zoomIn = useCallback(() => zoomBy(factor), [zoomBy, factor])
  const zoomOut = useCallback(() => zoomBy(1 / factor), [zoomBy, factor])

  const reset = useCallback(() => {
    if (!latest.current.bounds) return
    latest.current.onChange(null)
  }, [])

  const drag = useRef<{
    pointerId: number
    clientX: number
    origin: TimeDomain
    moved: boolean
  } | null>(null)

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    const { view } = latest.current
    /* Primary button / single touch only: a right click or a second finger
       must not start a pan. */
    if (!view || event.button !== 0 || !event.isPrimary) return
    if (!(plotWidth(event.currentTarget) > 0)) return
    drag.current = { pointerId: event.pointerId, clientX: event.clientX, origin: view, moved: false }
  }

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const current = drag.current
    const { bounds } = latest.current
    if (!current || !bounds || event.pointerId !== current.pointerId) return
    const dx = event.clientX - current.clientX
    if (!current.moved) {
      /* Ignore the jitter of a plain click, so clicking the plot never pans. */
      if (Math.abs(dx) < DRAG_THRESHOLD_PX) return
      current.moved = true
      event.currentTarget.setPointerCapture?.(event.pointerId)
    }
    const width = plotWidth(event.currentTarget)
    if (!(width > 0)) return
    /* Dragging right pulls earlier time into view. */
    const deltaMs = (-dx / width) * (current.origin[1] - current.origin[0])
    apply(panDomain(current.origin, deltaMs, bounds))
  }

  const endDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const current = drag.current
    if (!current) return
    drag.current = null
    if (current.moved && event.currentTarget.hasPointerCapture?.(current.pointerId)) {
      event.currentTarget.releasePointerCapture(current.pointerId)
    }
  }

  useEffect(() => {
    if (!surface) return
    const onWheel = (event: WheelEvent) => {
      /* Ctrl/⌘ + wheel is browser zoom, not chart zoom. */
      if (event.ctrlKey || event.metaKey || event.deltaY === 0) return
      const { view, bounds } = latest.current
      if (!view || !bounds) return
      event.preventDefault()
      const rect = surface.getBoundingClientRect()
      const width = rect.width - insets.left - insets.right
      const ratio =
        width > 0 ? clamp((event.clientX - rect.left - insets.left) / width, 0, 1) : 0.5
      const anchor = view[0] + ratio * (view[1] - view[0])
      const step = event.deltaY < 0 ? factor : 1 / factor
      apply(zoomDomain(view, step, anchor, bounds))
    }
    surface.addEventListener('wheel', onWheel, { passive: false })
    return () => surface.removeEventListener('wheel', onWheel)
  }, [surface, apply, factor, insets.left, insets.right])

  return {
    surfaceRef,
    zoomed: isZoomed(view, bounds),
    canZoomIn: canZoomIn(view, bounds),
    zoomIn,
    zoomOut,
    reset,
    handlers: {
      onPointerDown,
      onPointerMove,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
      onDoubleClick: reset,
    },
  }
}
