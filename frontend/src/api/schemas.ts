import { z } from 'zod'

/**
 * Runtime schemas for the backend contract in `docs/api-contract.md`.
 *
 * Parsing is tolerant (missing keys are coerced) but the resulting *types* are
 * strict, so a drifting backend fails loudly instead of rendering `undefined`.
 */

const nullableNumber = z
  .number()
  .nullish()
  .transform((value) => value ?? null)

const nullableString = z
  .string()
  .nullish()
  .transform((value) => value ?? null)

const isoDate = z.string()

export const currentWeatherSchema = z.object({
  city: z.string(),
  latitude: nullableNumber,
  longitude: nullableNumber,
  temperature: nullableNumber,
  apparent_temperature: nullableNumber,
  humidity: nullableNumber,
  pressure: nullableNumber,
  wind_speed: nullableNumber,
  wind_direction: nullableNumber,
  precipitation: nullableNumber,
  cloud_cover: nullableNumber,
  weather_code: nullableNumber,
  observed_at: isoDate,
})
export type CurrentWeather = z.infer<typeof currentWeatherSchema>

export const weatherPointSchema = currentWeatherSchema
export type WeatherPoint = z.infer<typeof weatherPointSchema>

export const predictionSchema = z.object({
  city: z.string(),
  source_timestamp: isoDate,
  prediction_timestamp: isoDate,
  horizon_hours: z
    .number()
    .optional()
    .transform((value) => value ?? 1),
  predicted_temperature: nullableNumber,
  predicted_rain: nullableNumber,
  observed_temperature: nullableNumber,
  /* Contract 2: prediction intervals are additive; legacy docs carry null. */
  temp_lower: nullableNumber,
  temp_upper: nullableNumber,
  interval_level: nullableNumber,
  temp_model_name: nullableString,
  temp_model_version: nullableString,
  rain_model_name: nullableString,
  rain_model_version: nullableString,
})
export type Prediction = z.infer<typeof predictionSchema>

export const metricsSchema = z.object({
  mae: nullableNumber,
  rmse: nullableNumber,
  bias: nullableNumber,
  n: z
    .number()
    .optional()
    .transform((value) => value ?? 0),
})
export type Metrics = z.infer<typeof metricsSchema>

export const seriesPointSchema = z.object({
  timestamp: isoDate,
  observed: nullableNumber,
  model: nullableNumber,
  aemet: nullableNumber,
  residual_model: nullableNumber,
  residual_aemet: nullableNumber,
})
export type SeriesPoint = z.infer<typeof seriesPointSchema>

export const benchmarkSchema = z.object({
  city: z.string(),
  hours: z.number(),
  generated_at: isoDate,
  aemet: z.object({
    available: z.boolean(),
    error: nullableString,
    issued_at: nullableString,
  }),
  series: z.array(seriesPointSchema),
  metrics: z.object({ model: metricsSchema, aemet: metricsSchema }),
})
export type Benchmark = z.infer<typeof benchmarkSchema>

export const benchmarkSummarySchema = z.object({
  generated_at: isoDate,
  aemet_configured: z.boolean(),
  cities: z.array(
    z.object({
      city: z.string(),
      n: z.number(),
      model: metricsSchema,
      aemet: metricsSchema,
    }),
  ),
})
export type BenchmarkSummary = z.infer<typeof benchmarkSummarySchema>

export const registryModelSchema = z.object({
  name: z.string(),
  version: z.string(),
  target: z.string(),
  horizon_hours: z.number(),
  created_at: nullableString,
  // The registry stores artefacts without metrics/stage, so the API sends an
  // explicit `null` for both; `nullish()` accepts it as well as a missing key.
  // Contract 3: the honest-metric keys are additive and nullable — a registry
  // row written before the temporal split simply omits them.
  metrics: z
    .object({
      rmse: nullableNumber,
      mae: nullableNumber,
      r2: nullableNumber,
      auc_roc: nullableNumber,
      auc_pr: nullableNumber,
      persistence_rmse: nullableNumber,
      climatology_rmse: nullableNumber,
      skill_score: nullableNumber,
      brier: nullableNumber,
      persistence_brier: nullableNumber,
      prevalence: nullableNumber,
      coverage: nullableNumber,
    })
    .partial()
    .nullish()
    .transform((value) => value ?? {}),
  split: z
    .object({
      kind: z.string(),
      train_end: nullableString,
      val_end: nullableString,
      test_start: nullableString,
    })
    .partial()
    .nullish()
    .transform((value) => value ?? null),
  interval: z
    .object({
      level: nullableNumber,
      lower_offset: nullableNumber,
      upper_offset: nullableNumber,
    })
    .partial()
    .nullish()
    .transform((value) => value ?? null),
  commit: nullableString,
  stage: z
    .string()
    .nullish()
    .transform((value) => value ?? 'none'),
})
export type RegistryModel = z.infer<typeof registryModelSchema>

export const modelsResponseSchema = z.object({
  count: z.number(),
  models: z.array(registryModelSchema),
})
export type ModelsResponse = z.infer<typeof modelsResponseSchema>

export const stationsResponseSchema = z.object({
  count: z.number(),
  stations: z.array(currentWeatherSchema),
})
export type StationsResponse = z.infer<typeof stationsResponseSchema>

/**
 * `GET /api/map/stations` — GeoJSON FeatureCollection of the latest observation
 * per city. Coordinates are `[longitude, latitude]`.
 */
export const stationFeatureSchema = z.object({
  type: z.literal('Feature'),
  geometry: z.object({
    type: z.literal('Point'),
    coordinates: z.tuple([z.number(), z.number()]),
  }),
  properties: z.object({
    city: z.string(),
    temperature: nullableNumber,
    apparent_temperature: nullableNumber,
    relative_humidity: nullableNumber,
    wind_speed: nullableNumber,
    wind_direction: nullableNumber,
    precipitation: nullableNumber,
    weather_code: nullableNumber,
    observed_at: isoDate,
  }),
})
export type StationFeature = z.infer<typeof stationFeatureSchema>

export const stationCollectionSchema = z.object({
  type: z.literal('FeatureCollection'),
  features: z.array(stationFeatureSchema),
})
export type StationCollection = z.infer<typeof stationCollectionSchema>

/** `GET /api/weather/quality?days=7` — per-city data-quality meter (Contract 1). */
export const weatherQualitySchema = z.object({
  generated_at: isoDate,
  days: z.number(),
  cities: z.array(
    z.object({
      city: z.string(),
      expected_hours: z.number(),
      observed_hours: z.number(),
      completeness: z.number(),
      max_gap_hours: nullableNumber,
      null_rate: nullableNumber,
      last_observed_at: nullableString,
      age_hours: nullableNumber,
      status: z.enum(['ok', 'warn', 'bad']),
    }),
  ),
})
export type WeatherQuality = z.infer<typeof weatherQualitySchema>
export type CityQuality = WeatherQuality['cities'][number]

export const historyResponseSchema = z.object({
  city: z.string(),
  hours: z.number(),
  count: z.number(),
  points: z.array(weatherPointSchema),
})
export type HistoryResponse = z.infer<typeof historyResponseSchema>

export const latestPredictionsSchema = z.object({
  count: z.number(),
  generated_at: nullableString,
  predictions: z.array(predictionSchema),
})
export type LatestPredictions = z.infer<typeof latestPredictionsSchema>

export const statsResponseSchema = z.object({
  city: z.string(),
  hours: z.number(),
  count: z.number(),
  temperature: z.object({ avg: nullableNumber, min: nullableNumber, max: nullableNumber }),
  humidity: z.object({ avg: nullableNumber, min: nullableNumber, max: nullableNumber }),
  pressure: z.object({ avg: nullableNumber, min: nullableNumber, max: nullableNumber }),
  wind_speed: z.object({ avg: nullableNumber, min: nullableNumber, max: nullableNumber }),
  precipitation_total: nullableNumber,
  start: nullableString,
  end: nullableString,
})
export type WeatherStats = z.infer<typeof statsResponseSchema>

export const healthSchema = z.object({
  status: z.string(),
  service: z.string(),
  version: z.string(),
  time: z.string(),
})
export type Health = z.infer<typeof healthSchema>

/** Station registry entry, as served by the `cities` collection. */
export const citySchema = z.object({
  name: z.string(),
  latitude: z.number(),
  longitude: z.number(),
})
export type City = z.infer<typeof citySchema>

/** Result of the Open-Meteo geocoding proxy. */
export const geoResultSchema = z.object({
  name: z.string(),
  latitude: z.number(),
  longitude: z.number(),
  country: nullableString,
  admin1: nullableString,
})
export type GeoResult = z.infer<typeof geoResultSchema>

export const geoSearchResponseSchema = z.object({
  results: z.array(geoResultSchema),
})
export type GeoSearchResponse = z.infer<typeof geoSearchResponseSchema>
