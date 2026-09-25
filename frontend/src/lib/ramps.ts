import { readCssVar } from '@/lib/chart-theme'

/**
 * Data colour ramps (design token family `--temp-1..7`, `--rain-1..5`,
 * `--wind-1..4`).
 *
 * The ramp is purely quantitative: it encodes magnitude and never stands in
 * for a series identity (`--obs`, `--model`, `--aemet`) or a semantic colour
 * (`--ok`, `--warn`, `--bad`). Values are quantised into the discrete token
 * steps so the map, the SVG charts and the CSS legend all agree.
 */

export type RampName = 'temp' | 'rain' | 'wind'

export const RAMP_STEPS: Record<RampName, number> = { temp: 7, rain: 5, wind: 4 }

/** Reference domains for the national map. Kept out of the components. */
export const RAMP_DOMAINS: Record<RampName, readonly [number, number]> = {
  temp: [-5, 40], // °C
  rain: [0, 20], // mm
  wind: [0, 60], // km/h
}

/** Mirrors the CSS tokens; used only when the property cannot be resolved. */
const RAMP_FALLBACK: Record<RampName, readonly string[]> = {
  temp: ['#0d0887', '#5402a3', '#8b0aa5', '#b73779', '#de645a', '#f98f36', '#f0f921'],
  rain: ['#c9e9f2', '#8fcbe6', '#4ea3d4', '#2373b0', '#123f73'],
  wind: ['#aeb8c8', '#8593a8', '#5b6b85', '#33445f'],
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

export function rampDomain(name: RampName): readonly [number, number] {
  return RAMP_DOMAINS[name]
}

/**
 * Quantise a value into a 0-based ramp step for the given domain.
 *
 * `null`, `undefined`, `NaN` and values outside the domain clamp to the nearest
 * step (never throw, never leave a hole in a data-driven map expression).
 */
export function rampStep(
  name: RampName,
  value: number | null | undefined,
  domain: readonly [number, number] = RAMP_DOMAINS[name],
): number {
  const steps = RAMP_STEPS[name]
  const [min, max] = domain
  if (typeof value !== 'number' || !Number.isFinite(value) || max === min) return 0
  const ratio = clamp((value - min) / (max - min), 0, 1)
  return Math.min(steps - 1, Math.max(0, Math.floor(ratio * steps)))
}

/** Interior step boundaries, for MapLibre `['step', ...]` expressions. */
export function rampBreaks(
  name: RampName,
  domain: readonly [number, number] = RAMP_DOMAINS[name],
): number[] {
  const steps = RAMP_STEPS[name]
  const [min, max] = domain
  return Array.from({ length: steps - 1 }, (_, index) => min + ((max - min) * (index + 1)) / steps)
}

/** CSS custom property for a step — the right value for DOM and SVG `var()`. */
export function rampToken(name: RampName, step: number): string {
  const index = clamp(Math.round(step), 0, RAMP_STEPS[name] - 1)
  return `var(--${name}-${index + 1})`
}

/**
 * Literal colour for MapLibre paint and SVG presentation attributes, which
 * cannot resolve `var()`. Reads the token sheet; falls back when detached.
 */
export function rampHex(name: RampName, step: number): string {
  const index = clamp(Math.round(step), 0, RAMP_STEPS[name] - 1)
  return readCssVar(`--${name}-${index + 1}`, RAMP_FALLBACK[name][index])
}

/** Convenience: value + domain -> CSS token. */
export function rampColor(
  name: RampName,
  value: number | null | undefined,
  domain?: readonly [number, number],
): string {
  return rampToken(name, rampStep(name, value, domain))
}

/** Convenience: value + domain -> literal colour. */
export function rampColorHex(
  name: RampName,
  value: number | null | undefined,
  domain?: readonly [number, number],
): string {
  return rampHex(name, rampStep(name, value, domain))
}
