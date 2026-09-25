import { describe, expect, it } from 'vitest'

import {
  benchmarkSchema,
  currentWeatherSchema,
  latestPredictionsSchema,
  modelsResponseSchema,
  predictionSchema,
  registryModelSchema,
  stationCollectionSchema,
  stationsResponseSchema,
  weatherQualitySchema,
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

describe('weatherQualitySchema (Contract 1)', () => {
  it('parses the frozen contract payload', () => {
    const parsed = weatherQualitySchema.parse({
      generated_at: '2026-09-25T12:00:00Z',
      days: 7,
      cities: [
        {
          city: 'Madrid',
          expected_hours: 168,
          observed_hours: 165,
          completeness: 0.982,
          max_gap_hours: 3,
          null_rate: 0.012,
          last_observed_at: '2026-09-25T11:00:00Z',
          age_hours: 1,
          status: 'ok',
        },
      ],
    })
    expect(parsed.cities[0].status).toBe('ok')
    expect(parsed.cities[0].completeness).toBeCloseTo(0.982)
  })

  it('rejects an unknown status', () => {
    const result = weatherQualitySchema.safeParse({
      generated_at: '2026-09-25T12:00:00Z',
      days: 7,
      cities: [{ city: 'X', expected_hours: 1, observed_hours: 1, completeness: 1, status: 'meh' }],
    })
    expect(result.success).toBe(false)
  })

  it('coerces nullable per-city fields to null', () => {
    const parsed = weatherQualitySchema.parse({
      generated_at: '2026-09-25T12:00:00Z',
      days: 7,
      cities: [{ city: 'X', expected_hours: 1, observed_hours: 0, completeness: 0, status: 'bad' }],
    })
    expect(parsed.cities[0].max_gap_hours).toBeNull()
    expect(parsed.cities[0].last_observed_at).toBeNull()
  })
})

describe('stationCollectionSchema (GET /map/stations)', () => {
  it('parses a GeoJSON FeatureCollection', () => {
    const parsed = stationCollectionSchema.parse({
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [-3.7038, 40.4168] },
          properties: {
            city: 'Madrid',
            temperature: 24.3,
            apparent_temperature: 24.8,
            relative_humidity: 61,
            wind_speed: 14.2,
            wind_direction: 210,
            precipitation: 0,
            weather_code: 0,
            observed_at: '2026-09-25T11:00:00Z',
          },
        },
      ],
    })
    expect(parsed.features[0].properties.city).toBe('Madrid')
    expect(parsed.features[0].geometry.coordinates).toEqual([-3.7038, 40.4168])
  })

  it('coerces missing observation values to null', () => {
    const parsed = stationCollectionSchema.parse({
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [0, 0] },
          properties: { city: 'X', observed_at: '2026-09-25T11:00:00Z' },
        },
      ],
    })
    expect(parsed.features[0].properties.temperature).toBeNull()
    expect(parsed.features[0].properties.weather_code).toBeNull()
  })
})

describe('prediction intervals (Contract 2)', () => {
  it('defaults legacy prediction fields to null', () => {
    const parsed = predictionSchema.parse({
      city: 'Madrid',
      source_timestamp: '2026-09-25T11:00:00Z',
      prediction_timestamp: '2026-09-25T12:00:00Z',
    })
    expect(parsed.temp_lower).toBeNull()
    expect(parsed.temp_upper).toBeNull()
    expect(parsed.interval_level).toBeNull()
  })

  it('keeps interval values when present', () => {
    const parsed = predictionSchema.parse({
      city: 'Madrid',
      source_timestamp: '2026-09-25T11:00:00Z',
      prediction_timestamp: '2026-09-25T12:00:00Z',
      predicted_temperature: 22,
      temp_lower: 20.1,
      temp_upper: 24.1,
      interval_level: 0.8,
    })
    expect(parsed.temp_lower).toBeCloseTo(20.1)
    expect(parsed.interval_level).toBe(0.8)
  })
})

describe('registry honest metrics (Contract 3)', () => {
  it('parses the additive metric blocks', () => {
    const parsed = registryModelSchema.parse({
      name: 'temp_prediction_1h_GradientBoostedTrees',
      version: '20260925_020000',
      target: 'temperature',
      horizon_hours: 1,
      created_at: '2026-09-25T02:00:00Z',
      metrics: {
        rmse: 1.44,
        persistence_rmse: 3.05,
        climatology_rmse: 3.41,
        skill_score: 0.53,
        coverage: 0.79,
      },
      split: { kind: 'temporal', test_start: '2026-09-01T00:00:00Z' },
      interval: { level: 0.8, lower_offset: -1.9, upper_offset: 2.1 },
      commit: 'abc1234def',
      stage: 'production',
    })
    expect(parsed.metrics.skill_score).toBeCloseTo(0.53)
    expect(parsed.metrics.coverage).toBeCloseTo(0.79)
    expect(parsed.split?.kind).toBe('temporal')
    expect(parsed.interval?.level).toBe(0.8)
    expect(parsed.commit).toBe('abc1234def')
  })

  it('defaults the new blocks to null on a legacy row', () => {
    const parsed = registryModelSchema.parse({
      name: 'temp_prediction_1h_GradientBoostedTrees',
      version: '20260923_020000',
      target: 'temperature',
      horizon_hours: 1,
    })
    expect(parsed.split).toBeNull()
    expect(parsed.interval).toBeNull()
    expect(parsed.commit).toBeNull()
    expect(parsed.metrics.skill_score).toBeUndefined()
  })
})
