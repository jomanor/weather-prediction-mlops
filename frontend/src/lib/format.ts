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

/** Map a value onto 0..1 across a range, guarding against a zero-width range. */
export function normalise(value: number, min: number, max: number): number {
  if (max === min) return 0.5
  return clamp((value - min) / (max - min), 0, 1)
}

/** "40,47° N" / "3,56° S" — ISO 6709 hemisphere letters (mono-safe, unambiguous). */
export function formatLatitude(value: number | null | undefined, digits = 2): string {
  if (!isNum(value)) return EMPTY
  const hemisphere = value >= 0 ? 'N' : 'S'
  return `${fixed(Math.abs(value), digits)}° ${hemisphere}`
}

/** "3,56° E" / "0,48° W" — ISO 6709 hemisphere letters. */
export function formatLongitude(value: number | null | undefined, digits = 2): string {
  if (!isNum(value)) return EMPTY
  const hemisphere = value >= 0 ? 'E' : 'W'
  return `${fixed(Math.abs(value), digits)}° ${hemisphere}`
}
