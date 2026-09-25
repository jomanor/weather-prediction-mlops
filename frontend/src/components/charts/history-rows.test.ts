import { describe, expect, it } from 'vitest'

import { currentWeatherSchema, predictionSchema, type Prediction, type WeatherPoint } from '@/api/schemas'
import { buildHistoryChartData } from '@/components/charts/history-rows'

const TS = '2026-09-25T10:00:00Z'

function point(observed_at: string, temperature: number | null): WeatherPoint {
  return currentWeatherSchema.parse({ city: 'Madrid', observed_at, temperature })
}

function prediction(input: {
  prediction_timestamp: string
  predicted_temperature: number | null
  temp_lower?: number | null
  temp_upper?: number | null
  interval_level?: number | null
}): Prediction {
  return predictionSchema.parse({
    city: 'Madrid',
    source_timestamp: '2026-09-25T09:00:00Z',
    ...input,
  })
}

describe('buildHistoryChartData', () => {
  it('merges an observation and a prediction on the same hour into one row', () => {
    const { rows, hasModel, hasInterval } = buildHistoryChartData(
      [point(TS, 21)],
      [prediction({ prediction_timestamp: TS, predicted_temperature: 22 })],
      'metric',
    )
    expect(rows).toHaveLength(1)
    expect(rows[0].temperature).toBeCloseTo(21)
    expect(rows[0].model).toBeCloseTo(22)
    expect(hasModel).toBe(true)
    // Legacy prediction: no lower/upper, so no band.
    expect(hasInterval).toBe(false)
    expect(rows[0].band_base).toBeNull()
    expect(rows[0].band_size).toBeNull()
  })

  it('derives band_base and band_size from temp_lower/temp_upper', () => {
    const { rows, hasInterval } = buildHistoryChartData(
      [point(TS, 21)],
      [
        prediction({
          prediction_timestamp: TS,
          predicted_temperature: 22.1,
          temp_lower: 20.1,
          temp_upper: 24.1,
          interval_level: 0.8,
        }),
      ],
      'metric',
    )
    expect(hasInterval).toBe(true)
    expect(rows[0].band_base).toBeCloseTo(20.1)
    expect(rows[0].band_size).toBeCloseTo(4.0)
  })

  it('clamps an inverted interval to zero width instead of drawing backwards', () => {
    const { rows } = buildHistoryChartData(
      [point(TS, 21)],
      [
        prediction({
          prediction_timestamp: TS,
          predicted_temperature: 22,
          temp_lower: 25,
          temp_upper: 20,
        }),
      ],
      'metric',
    )
    expect(rows[0].band_base).toBeCloseTo(25)
    expect(rows[0].band_size).toBe(0)
  })

  it('ignores a prediction without a predicted temperature', () => {
    const { rows, hasModel } = buildHistoryChartData(
      [point(TS, 21)],
      [prediction({ prediction_timestamp: '2026-09-25T11:00:00Z', predicted_temperature: null })],
      'metric',
    )
    expect(hasModel).toBe(false)
    expect(rows).toHaveLength(1)
  })

  it('converts interval bounds with the unit system', () => {
    const { rows } = buildHistoryChartData(
      [point(TS, 21)],
      [
        prediction({
          prediction_timestamp: TS,
          predicted_temperature: 22,
          temp_lower: 0,
          temp_upper: 10,
        }),
      ],
      'imperial',
    )
    expect(rows[0].band_base).toBeCloseTo(32)
    expect(rows[0].band_size).toBeCloseTo(18)
  })

  it('sorts the merged timeline ascending (oldest first)', () => {
    const { rows } = buildHistoryChartData(
      [point('2026-09-25T12:00:00Z', 23), point('2026-09-25T10:00:00Z', 21)],
      [],
      'metric',
    )
    expect(rows.map((row) => row.ts)).toEqual([
      new Date('2026-09-25T10:00:00Z').getTime(),
      new Date('2026-09-25T12:00:00Z').getTime(),
    ])
  })
})
