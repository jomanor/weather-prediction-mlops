import { describe, expect, it } from 'vitest'

import { predictionSchema, type Prediction } from '@/api/schemas'
import { forecastTargetMs, latestRun, pickHorizon } from '@/lib/predictions'

function prediction(input: {
  source_timestamp: string
  prediction_timestamp: string
  horizon_hours: number
  predicted_temperature?: number
}): Prediction {
  return predictionSchema.parse({ city: 'Madrid', ...input })
}

describe('latestRun', () => {
  it('returns only the rows with the maximum source_timestamp', () => {
    const older = prediction({
      source_timestamp: '2026-09-25T08:00:00Z',
      prediction_timestamp: '2026-09-25T08:05:00Z',
      horizon_hours: 1,
    })
    const newer = prediction({
      source_timestamp: '2026-09-25T10:00:00Z',
      prediction_timestamp: '2026-09-25T10:05:00Z',
      horizon_hours: 1,
    })
    expect(latestRun([older, newer])).toEqual([newer])
  })

  it('keeps one row per horizon when the newest run has duplicates', () => {
    const first = prediction({
      source_timestamp: '2026-09-25T10:00:00Z',
      prediction_timestamp: '2026-09-25T10:05:00Z',
      horizon_hours: 3,
      predicted_temperature: 1,
    })
    const duplicate = prediction({
      source_timestamp: '2026-09-25T10:00:00Z',
      prediction_timestamp: '2026-09-25T10:05:00Z',
      horizon_hours: 3,
      predicted_temperature: 2,
    })
    const other = prediction({
      source_timestamp: '2026-09-25T10:00:00Z',
      prediction_timestamp: '2026-09-25T10:05:00Z',
      horizon_hours: 1,
    })
    const run = latestRun([first, duplicate, other])
    expect(run).toHaveLength(2)
    expect(run.map((row) => row.horizon_hours)).toEqual([1, 3])
    expect(run[1]).toBe(first)
  })

  it('returns nothing when no source_timestamp parses', () => {
    expect(latestRun([])).toEqual([])
  })
})

describe('forecastTargetMs', () => {
  it('adds the horizon to the source observation instant', () => {
    const row = prediction({
      source_timestamp: '2026-09-25T09:00:00Z',
      prediction_timestamp: '2026-09-25T09:05:00Z',
      horizon_hours: 6,
    })
    expect(forecastTargetMs(row)).toBe(new Date('2026-09-25T15:00:00Z').getTime())
  })
})

describe('pickHorizon', () => {
  it('returns the exact horizon row', () => {
    const one = prediction({
      source_timestamp: '2026-09-25T09:00:00Z',
      prediction_timestamp: '2026-09-25T09:05:00Z',
      horizon_hours: 1,
    })
    const three = prediction({
      source_timestamp: '2026-09-25T09:00:00Z',
      prediction_timestamp: '2026-09-25T09:05:00Z',
      horizon_hours: 3,
    })
    expect(pickHorizon([one, three], 3)).toBe(three)
  })

  it('falls back to the newest prediction_timestamp when the horizon is missing', () => {
    const older = prediction({
      source_timestamp: '2026-09-25T09:00:00Z',
      prediction_timestamp: '2026-09-25T09:01:00Z',
      horizon_hours: 1,
    })
    const newer = prediction({
      source_timestamp: '2026-09-25T09:00:00Z',
      prediction_timestamp: '2026-09-25T09:06:00Z',
      horizon_hours: 6,
    })
    expect(pickHorizon([older, newer], 3)).toBe(newer)
  })

  it('returns undefined for an empty run', () => {
    expect(pickHorizon([], 3)).toBeUndefined()
  })
})
