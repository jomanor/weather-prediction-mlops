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
    ok: readVar('--ok', FALLBACK.ok),
    warn: readVar('--warn', FALLBACK.warn),
    bad: readVar('--bad', FALLBACK.bad),
    accent: readVar('--accent', FALLBACK.accent),
  }
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
