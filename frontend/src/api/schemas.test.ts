import { describe, expect, it } from 'vitest'

import {
  benchmarkSchema,
  currentWeatherSchema,
  latestPredictionsSchema,
  modelsResponseSchema,
  stationsResponseSchema,
} from '@/api/schemas'

describe('currentWeatherSchema', () => {
  it('accepts a full contract payload', () => {
    const parsed = currentWeatherSchema.parse({
      city: 'Madrid',
      latitude: 40.4168,
      longitude: -3.7038,
      temperature: 24.3,
      apparent_temperature: 24.8,
      humidity: 61,
      pressure: 1014.2,
      wind_speed: 14.2,
      wind_direction: 210,
      precipitation: 0,
      cloud_cover: 15,
      weather_code: 0,
      observed_at: '2026-09-23T14:00:00Z',
    })
    expect(parsed.city).toBe('Madrid')
    expect(parsed.temperature).toBe(24.3)
  })

  it('coerces missing nullable fields to null instead of undefined', () => {
    const parsed = currentWeatherSchema.parse({
      city: 'Vigo',
      observed_at: '2026-09-23T14:00:00Z',
    })
    expect(parsed.temperature).toBeNull()
    expect(parsed.wind_speed).toBeNull()
    expect(parsed.weather_code).toBeNull()
  })

  it('rejects a non-numeric temperature', () => {
    const result = currentWeatherSchema.safeParse({
      city: 'Vigo',
      temperature: 'warm',
      observed_at: '2026-09-23T14:00:00Z',
    })
    expect(result.success).toBe(false)
  })
})

describe('benchmarkSchema', () => {
  it('parses the AEMET-unavailable shape with null series values', () => {
    const parsed = benchmarkSchema.parse({
      city: 'Madrid',
      hours: 24,
      generated_at: '2026-09-23T14:00:00Z',
      aemet: { available: false, error: 'AEMET_API_KEY no configurada', issued_at: null },
      series: [
        {
          timestamp: '2026-09-23T14:00:00Z',
          observed: 24.1,
          model: 23.6,
          aemet: null,
          residual_model: -0.5,
          residual_aemet: null,
        },
      ],
      metrics: {
        model: { mae: 1.12, rmse: 1.44, bias: -0.08, n: 24 },
        aemet: { mae: null, rmse: null, bias: null, n: 0 },
      },
    })
    expect(parsed.aemet.available).toBe(false)
    expect(parsed.series[0].aemet).toBeNull()
    expect(parsed.metrics.aemet.mae).toBeNull()
  })
})

describe('stationsResponseSchema', () => {
  it('parses an empty network', () => {
    expect(stationsResponseSchema.parse({ count: 0, stations: [] }).stations).toEqual([])
  })
})

describe('latestPredictionsSchema', () => {
  it('fills defaults for optional numeric fields', () => {
    const parsed = latestPredictionsSchema.parse({
      count: 1,
      generated_at: null,
      predictions: [
        {
          city: 'Madrid',
          source_timestamp: '2026-09-23T13:00:00Z',
          prediction_timestamp: '2026-09-23T13:05:00Z',
        },
      ],
    })
    expect(parsed.predictions[0].horizon_hours).toBe(1)
    expect(parsed.predictions[0].predicted_temperature).toBeNull()
    expect(parsed.predictions[0].temp_model_name).toBeNull()
  })
})

describe('modelsResponseSchema', () => {
  it('tolerates a registry entry without metrics', () => {
    const parsed = modelsResponseSchema.parse({
      count: 1,
      models: [
        {
          name: 'temp_prediction_1h_GradientBoostedTrees',
          version: '20260923_020000',
          target: 'temperature',
          horizon_hours: 1,
          created_at: null,
        },
      ],
    })
    expect(parsed.models[0].metrics).toEqual({})
    expect(parsed.models[0].stage).toBe('none')
  })

  it('tolerates explicit null metrics and stage from the registry', () => {
    const parsed = modelsResponseSchema.parse({
      count: 1,
      models: [
        {
          name: 'rain_prediction_1h_GradientBoostedTrees',
          version: '20260925_183924',
          target: 'rain',
          horizon_hours: 1,
          created_at: '2026-09-25T18:39:24.858000Z',
          metrics: null,
          stage: null,
        },
      ],
    })
    expect(parsed.models[0].metrics).toEqual({})
    expect(parsed.models[0].stage).toBe('none')
  })
})
