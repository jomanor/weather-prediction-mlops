export type ThemeChoice = 'light' | 'dark' | 'system'

export interface ChartPalette {
  observed: string
  model: string
  aemet: string
  grid: string
  axis: string
  text: string
  textMuted: string
  panel: string
  line: string
  lineStrong: string
  ok: string
  warn: string
  bad: string
  accent: string
}

const FALLBACK: ChartPalette = {
  observed: '#f1f3f6',
  model: '#ab90fb',
  aemet: '#f0a92b',
  grid: 'rgba(233,236,240,0.07)',
  axis: '#232932',
  text: '#e9ecf0',
  textMuted: '#97a1ad',
  panel: '#13161a',
  line: '#232932',
  lineStrong: '#3a424e',
  ok: '#3ecf8e',
  warn: '#e0a11a',
  bad: '#f2604f',
  accent: '#48b7f3',
}

function readVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

/**
 * Resolve a CSS custom property to a literal value.
 *
 * MapLibre `paint`/`layout` values and SVG attributes cannot resolve `var()`,
 * so colours are read from the token sheet on the main thread and passed down
 * as literals. Components must never hardcode a hex; this is the bridge.
 */
export function readCssVar(name: string, fallback: string): string {
  return readVar(name, fallback)
}

/**
 * Chart colours are read from the CSS custom properties so the design tokens
 * stay the single source of truth for both the DOM and the SVG charts.
 */
export function readChartPalette(isDark: boolean): ChartPalette {
  void isDark
  return {
    observed: readVar('--obs', FALLBACK.observed),
    model: readVar('--model', FALLBACK.model),
    aemet: readVar('--aemet', FALLBACK.aemet),
    grid: readVar('--grid', FALLBACK.grid),
    axis: readVar('--line', FALLBACK.axis),
    text: readVar('--fg', FALLBACK.text),
    textMuted: readVar('--fg-3', FALLBACK.textMuted),
    panel: readVar('--panel', FALLBACK.panel),
    line: readVar('--line', FALLBACK.line),
    lineStrong: readVar('--line-strong', FALLBACK.lineStrong),
    ok: readVar('--ok', FALLBACK.ok),
    warn: readVar('--warn', FALLBACK.warn),
    bad: readVar('--bad', FALLBACK.bad),
    accent: readVar('--accent', FALLBACK.accent),
  }
}

/** One margin for line/area charts; bar charts add room for their labels. */
export const CHART_MARGIN = { top: 8, right: 10, bottom: 0, left: 0 } as const

/** Axis tick type shared by every chart, so the scale reads as one system. */
const TICK_FONT = 'JetBrains Mono Variable, monospace'

/** Stroke/tick props for a Recharts axis, read from the design tokens. */
export function chartAxisProps(palette: ChartPalette) {
  return {
    stroke: palette.axis,
    tick: { fill: palette.textMuted, fontSize: 10, fontFamily: TICK_FONT },
    tickLine: false as const,
  }
}

/** Rotated axis title: the unit is part of the scale, not a floating caption. */
export function chartAxisLabel(
  palette: ChartPalette,
  value: string,
  position: 'insideLeft' | 'insideRight' = 'insideLeft',
) {
  return {
    value,
    position,
    angle: -90,
    fill: palette.textMuted,
    fontSize: 10,
    fontFamily: TICK_FONT,
  }
}

/** Dotted horizontal grid, no vertical rules (time is read from the ticks). */
export function chartGridProps(palette: ChartPalette) {
  return { stroke: palette.grid, strokeDasharray: '2 4', vertical: false as const }
}

/** Vertical guide under the pointer, matching the axis hairline. */
export function chartLineCursor(palette: ChartPalette) {
  return { stroke: palette.axis, strokeDasharray: '3 3' }
}

/** Translucent column highlight for bar charts. */
export function chartBarCursor(palette: ChartPalette) {
  return { fill: palette.grid }
}

export const SERIES_LABELS = {
  observed: 'Observado',
  model: 'Spark GBT',
  aemet: 'AEMET',
} as const

/**
 * MapLibre `paint` values are the one documented exception to "no hardcoded
 * hex". The style spec accepts colour strings only — not CSS custom properties
 * — and the GL worker cannot resolve `getComputedStyle`, so the values cannot
 * be read from the DOM. They are kept here, beside the chart palette, so both
 * themes stay in one reviewable place.
 */
export const MAP_PAINT = {
  hillshadeShadow: { light: '#5b6470', dark: '#000000' },
  building: { light: '#cfd6dd', dark: '#1b222b' },
} as const
