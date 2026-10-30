import type { TimeDomain } from '@/hooks/useChartSync'
import { clamp } from '@/lib/format'

/**
 * Zoom/pan arithmetic for the time axis, kept pure so the interaction is
 * testable without a chart. Every function takes the *bounds* — the widest
 * window the user may reach (the page window or the data extent) — and never
 * returns a domain outside them, so a chart can never be panned into a gap.
 */

/** Shortest window the zoom can reach: two hours of observations. */
export const MIN_ZOOM_SPAN_MS = 2 * 3_600_000

/** Span divisor per zoom step (one wheel notch or one button press). */
export const ZOOM_FACTOR = 1.25

/** Float noise guard: sub-millisecond differences are not a zoom. */
const EPSILON_MS = 1

function span(domain: TimeDomain): number {
  return Math.abs(domain[1] - domain[0])
}

/** Widest of two domains; `null` means "no domain", not "zero width". */
export function unionDomain(
  a: TimeDomain | null,
  b: TimeDomain | null,
): TimeDomain | null {
  if (!a) return b
  if (!b) return a
  return [Math.min(a[0], b[0]), Math.max(a[1], b[1])]
}

/** True when the domain is narrower than the bounds, i.e. the user zoomed in. */
export function isZoomed(domain: TimeDomain | null, bounds: TimeDomain | null): boolean {
  if (!domain || !bounds) return false
  return span(domain) < span(bounds) - EPSILON_MS
}

/** True when there is room left before the minimum span is reached. */
export function canZoomIn(domain: TimeDomain | null, bounds: TimeDomain | null): boolean {
  if (!domain || !bounds) return false
  return span(domain) > Math.min(MIN_ZOOM_SPAN_MS, span(bounds)) + EPSILON_MS
}

/**
 * Fit `domain` inside `bounds`, growing it to the minimum span (around its
 * centre) when it is too narrow and capping it when it is wider than the
 * bounds. Degenerate bounds (`end <= start`) collapse the domain to the bound
 * edge — a chart with no time extent has nothing to zoom.
 */
export function clampDomain(domain: TimeDomain, bounds: TimeDomain): TimeDomain {
  const [boundsStart, boundsEnd] = bounds
  const boundsSpan = boundsEnd - boundsStart
  if (!(boundsSpan > 0)) return [boundsStart, boundsEnd]

  const width = clamp(span(domain), Math.min(MIN_ZOOM_SPAN_MS, boundsSpan), boundsSpan)
  const center = (domain[0] + domain[1]) / 2
  const start = clamp(center - width / 2, boundsStart, boundsEnd - width)
  return [start, start + width]
}

/**
 * Keep `anchor` (epoch ms, e.g. the pointer) at the same screen position while
 * the span changes. `factor > 1` zooms in; the anchor is clamped to the view so
 * a wheel over the axis gutter still zooms towards the nearest edge.
 */
export function zoomDomain(
  domain: TimeDomain,
  factor: number,
  anchor: number,
  bounds: TimeDomain,
): TimeDomain {
  const left = Math.min(domain[0], domain[1])
  const right = Math.max(domain[0], domain[1])
  const width = right - left
  if (!(factor > 0) || !(width > 0)) return clampDomain(domain, bounds)

  const ratio = clamp((anchor - left) / width, 0, 1)
  const nextWidth = width / factor
  const start = anchor - ratio * nextWidth
  return clampDomain([start, start + nextWidth], bounds)
}

/** Shift the window by `deltaMs`, keeping its width, without leaving `bounds`. */
export function panDomain(
  domain: TimeDomain,
  deltaMs: number,
  bounds: TimeDomain,
): TimeDomain {
  return clampDomain([domain[0] + deltaMs, domain[1] + deltaMs], bounds)
}

/** The un-zoomed view: always the whole of `bounds`, never a stale selection. */
export function resetDomain(bounds: TimeDomain): TimeDomain {
  return clampDomain([bounds[0], bounds[1]], bounds)
}
