export type UnitSystem = 'metric' | 'imperial'

/** Placeholder shown whenever the backend legitimately has no value. */
export const EMPTY = '—'

export function isNum(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function fixed(value: number, digits: number): string {
  return new Intl.NumberFormat('es-ES', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)
}

export function formatNumber(value: number | null | undefined, digits = 1): string {
  return isNum(value) ? fixed(value, digits) : EMPTY
}

export function formatInteger(value: number | null | undefined): string {
  return isNum(value) ? new Intl.NumberFormat('es-ES').format(value) : EMPTY
}

export function convertTemperature(celsius: number, units: UnitSystem): number {
  return units === 'imperial' ? celsius * 1.8 + 32 : celsius
}

export function formatTemperature(
  celsius: number | null | undefined,
  units: UnitSystem = 'metric',
  digits = 1,
): string {
  if (!isNum(celsius)) return EMPTY
  return `${fixed(convertTemperature(celsius, units), digits)}°`
}

export function temperatureUnit(units: UnitSystem): string {
  return units === 'imperial' ? '°F' : '°C'
}

export function windUnit(units: UnitSystem): string {
  return units === 'imperial' ? 'mph' : 'km/h'
}

/** Whole hours elapsed since an ISO timestamp, or null when absent/invalid. */
export function ageInHours(
  iso: string | null | undefined,
  now = Date.now(),
): number | null {
  if (!iso) return null
  const parsed = new Date(iso).getTime()
  if (!Number.isFinite(parsed)) return null
  return Math.max(0, (now - parsed) / 3_600_000)
}

/** Human age of a timestamp: "hace 34 min" / "hace 3 h". */
export function formatAgeHours(iso: string | null | undefined, now = Date.now()): string {
  const hours = ageInHours(iso, now)
  if (hours === null) return EMPTY
  if (hours < 1) return `hace ${formatNumber(Math.max(1, Math.round(hours * 60)), 0)} min`
  return `hace ${formatNumber(Math.round(hours), 0)} h`
}

export function formatWind(kmh: number | null | undefined, units: UnitSystem = 'metric'): string {
  if (!isNum(kmh)) return EMPTY
  const value = units === 'imperial' ? kmh * 0.621371 : kmh
  return `${fixed(value, 1)} ${units === 'imperial' ? 'mph' : 'km/h'}`
}

export function formatPressure(hpa: number | null | undefined, units: UnitSystem = 'metric'): string {
  if (!isNum(hpa)) return EMPTY
  return units === 'imperial' ? `${fixed(hpa * 0.02953, 2)} inHg` : `${fixed(hpa, 1)} hPa`
}

export function formatPrecipitation(
  mm: number | null | undefined,
  units: UnitSystem = 'metric',
): string {
  if (!isNum(mm)) return EMPTY
  return units === 'imperial' ? `${fixed(mm * 0.03937, 2)} in` : `${fixed(mm, 1)} mm`
}

export function formatPercent(value: number | null | undefined, digits = 0): string {
  return isNum(value) ? `${fixed(value, digits)}%` : EMPTY
}

export function formatSigned(value: number | null | undefined, digits = 2): string {
  if (!isNum(value)) return EMPTY
  const sign = value > 0 ? '+' : value < 0 ? '−' : ''
  return `${sign}${fixed(Math.abs(value), digits)}`
}

export function formatDate(iso: string | null | undefined, locale = 'es-ES'): string {
  if (!iso) return EMPTY
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return EMPTY
  return new Intl.DateTimeFormat(locale, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  }).format(date)
}

export function formatTime(iso: string | null | undefined, locale = 'es-ES'): string {
  if (!iso) return EMPTY
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return EMPTY
  return new Intl.DateTimeFormat(locale, { hour: '2-digit', minute: '2-digit' }).format(date)
}

export function formatDateTime(iso: string | null | undefined, locale = 'es-ES'): string {
  if (!iso) return EMPTY
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return EMPTY
  return new Intl.DateTimeFormat(locale, {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

/** Chart x-axis tick: epoch milliseconds -> "dd MMM HH:mm". */
export function formatDateTimeMs(ms: number, locale = 'es-ES'): string {
  return formatDateTime(new Date(ms).toISOString(), locale)
}

export function formatRelative(iso: string | null | undefined, locale = 'es-ES'): string {
  if (!iso) return EMPTY
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return EMPTY
  const diffMinutes = Math.round((date.getTime() - Date.now()) / 60_000)
  const formatter = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' })
  const abs = Math.abs(diffMinutes)
  if (abs < 60) return formatter.format(diffMinutes, 'minute')
  if (abs < 60 * 24) return formatter.format(Math.round(diffMinutes / 60), 'hour')
  return formatter.format(Math.round(diffMinutes / (60 * 24)), 'day')
}

const COMPASS = [
  'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
  'S', 'SSO', 'SO', 'OSO', 'O', 'ONO', 'NO', 'NNO',
]

export function compassPoint(degrees: number | null | undefined): string {
  if (!isNum(degrees)) return EMPTY
  const index = Math.round((((degrees % 360) + 360) % 360) / 22.5) % 16
  return COMPASS[index]
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/* ------------------------------------------------------------------------- *
 * Canonical precision and chart metric formatting (docs/app-rethink.md §3).
 *
 * One number of decimals per quantity, held everywhere: temperature 1 dp,
 * error 2 dp, skill 2 dp, percentages 0–1 dp. Charts round their axis ticks
 * (the unit and the precision live in the readout/tooltip, once) but never
 * invent their own precision.
 * ------------------------------------------------------------------------- */

export const PRECISION = {
  temperature: 1,
  error: 2,
  skill: 2,
  /** Humidity, coverage, completeness: whole percent. */
  percent: 0,
  /** Probability / confidence / interval level: 1 dp. */
  probability: 1,
  wind: 1,
  precipitation: 1,
  pressure: 1,
} as const

/** The metrics the charts can plot. */
export type Metric = 'temperature' | 'wind' | 'precipitation'

export const METRIC_DIGITS: Record<Metric, number> = {
  temperature: PRECISION.temperature,
  wind: PRECISION.wind,
  precipitation: PRECISION.precipitation,
}

/** Axis ticks round for legibility; the axis title carries the unit. */
export const METRIC_TICK_DIGITS: Record<Metric, number> = {
  temperature: 0,
  wind: 0,
  precipitation: 1,
}

export function precipitationUnit(units: UnitSystem = 'metric'): string {
  return units === 'imperial' ? 'in' : 'mm'
}

/** Unit label of a metric in the active unit system: "°C" / "km/h" / "mm". */
export function metricUnit(metric: Metric, units: UnitSystem = 'metric'): string {
  if (metric === 'temperature') return temperatureUnit(units)
  if (metric === 'wind') return windUnit(units)
  return precipitationUnit(units)
}

/**
 * Value + unit for a tooltip, axis label or readout: "21,4 °C".
 *
 * The value is expected already in the display unit system — chart rows convert
 * when they are built, so render-time formatting never converts twice.
 */
export function formatMetric(
  value: number | null | undefined,
  metric: Metric,
  units: UnitSystem = 'metric',
  digits: number = METRIC_DIGITS[metric],
): string {
  return isNum(value) ? `${fixed(value, digits)} ${metricUnit(metric, units)}` : EMPTY
}

/** Axis tick without its unit: "21" / "1,2". */
export function formatMetricTick(
  value: number | null | undefined,
  metric: Metric,
  digits: number = METRIC_TICK_DIGITS[metric],
): string {
  return isNum(value) ? fixed(value, digits) : EMPTY
}

/** Signed model error with its unit at the canonical 2 dp: "+0,42 °C". */
export function formatError(
  value: number | null | undefined,
  unit: string,
  digits: number = PRECISION.error,
): string {
  return isNum(value) ? `${formatSigned(value, digits)} ${unit}` : EMPTY
}

/** Chart data key → metric: the one place the mapping lives. */
export function metricForSeries(dataKey: string | number | undefined): Metric {
  if (dataKey === 'precipitation') return 'precipitation'
  if (dataKey === 'wind') return 'wind'
  return 'temperature'
}
