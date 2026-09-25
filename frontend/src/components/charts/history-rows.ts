import type { Prediction, WeatherPoint } from '@/api/schemas'
import { convertTemperature, isNum, type UnitSystem } from '@/lib/format'

/** One merged timeline row: observations and model predictions share the hour. */
export interface HistoryChartRow {
  ts: number
  temperature: number | null
  precipitation: number | null
  wind: number | null
  model: number | null
  /** Interval band, stacked: transparent base + model-tinted size. */
  band_base: number | null
  band_size: number | null
}

export interface HistoryChartData {
  rows: HistoryChartRow[]
  temperatureDomain: [number, number]
  precipitationMax: number
  windDomain: [number, number]
  hasInterval: boolean
  hasModel: boolean
}

/**
 * Merges observed history and model predictions onto one ascending timeline.
 *
 * The merge is per timestamp on purpose: interleaving observed and predicted
 * rows would break every line at the alternating nulls (`connectNulls` is off),
 * so an hour bucket carries both the observation and the model value.
 */
export function buildHistoryChartData(
  points: WeatherPoint[],
  predictions: Prediction[],
  units: UnitSystem,
): HistoryChartData {
  const rows = new globalThis.Map<number, HistoryChartRow>()
  const bucket = (ts: number): HistoryChartRow => {
    const existing = rows.get(ts)
    if (existing) return existing
    const created: HistoryChartRow = {
      ts,
      temperature: null,
      precipitation: null,
      wind: null,
      model: null,
      band_base: null,
      band_size: null,
    }
    rows.set(ts, created)
    return created
  }

  for (const point of points) {
    const ts = new Date(point.observed_at).getTime()
    if (!Number.isFinite(ts)) continue
    const row = bucket(ts)
    if (isNum(point.temperature)) row.temperature = convertTemperature(point.temperature, units)
    if (isNum(point.precipitation)) row.precipitation = point.precipitation
    if (isNum(point.wind_speed)) {
      row.wind = units === 'imperial' ? point.wind_speed * 0.621371 : point.wind_speed
    }
  }

  let hasInterval = false
  let hasModel = false
  for (const prediction of predictions) {
    const ts = new Date(prediction.prediction_timestamp).getTime()
    if (!Number.isFinite(ts) || !isNum(prediction.predicted_temperature)) continue
    const row = bucket(ts)
    row.model = convertTemperature(prediction.predicted_temperature, units)
    hasModel = true
    if (isNum(prediction.temp_lower) && isNum(prediction.temp_upper)) {
      const lower = convertTemperature(prediction.temp_lower, units)
      const upper = convertTemperature(prediction.temp_upper, units)
      row.band_base = lower
      /* Inverted intervals clamp to zero width rather than drawing backwards. */
      row.band_size = Math.max(0, upper - lower)
      hasInterval = true
    }
  }

  const sorted = [...rows.values()].sort((a, b) => a.ts - b.ts)

  const temperatureValues = sorted
    .flatMap((row) =>
      row.band_base !== null && row.band_size !== null
        ? [row.temperature, row.model, row.band_base, row.band_base + row.band_size]
        : [row.temperature, row.model],
    )
    .filter((value): value is number => value !== null)
  let temperatureDomain: [number, number] = [0, 1]
  if (temperatureValues.length) {
    const min = Math.min(...temperatureValues)
    const max = Math.max(...temperatureValues)
    const pad = Math.max((max - min) * 0.15, 1)
    temperatureDomain = [Math.floor(min - pad), Math.ceil(max + pad)]
  }

  const precipitationMax = Math.max(1, ...sorted.map((row) => row.precipitation ?? 0)) * 1.2

  const winds = sorted.map((row) => row.wind).filter((value): value is number => value !== null)
  let windDomain: [number, number] = [0, 1]
  if (winds.length) {
    const min = Math.min(...winds)
    const max = Math.max(...winds)
    const pad = Math.max((max - min) * 0.15, 1)
    windDomain = [Math.max(0, Math.floor(min - pad)), Math.ceil(max + pad)]
  }

  return { rows: sorted, temperatureDomain, precipitationMax, windDomain, hasInterval, hasModel }
}
