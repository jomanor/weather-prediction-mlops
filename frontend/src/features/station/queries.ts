import { z } from 'zod'

import { queryKeys, useApiQuery } from '@/api/queries'
import { predictionSchema, seriesResponseSchema, statsResponseSchema } from '@/api/schemas'

/** Station-scoped queries: the observation series, stats and predictions. */

/**
 * Exactly the columns this surface renders, as accepted by the backend's
 * `FIELDS_QUERY` (it rejects anything outside `WEATHER_FIELDS` with a 422):
 *
 * - hero card: `temperature`, `apparent_temperature`, `humidity`, `pressure`,
 *   `wind_speed`, `wind_direction`, `precipitation`, `cloud_cover`,
 *   `weather_code`, `is_day` (the icon reads the station's own day flag);
 * - chart: `temperature`, `precipitation`, `wind_speed`.
 *
 * The union is the whole contract today, so the projection saves nothing yet —
 * it is pinned so an 11th `weather_data` column cannot start arriving in a
 * payload nobody reads. `latitude`/`longitude`/`city`/`observed_at` are the
 * identity columns and are always projected by the API.
 */
export const STATION_SERIES_FIELDS = [
  'temperature',
  'apparent_temperature',
  'humidity',
  'pressure',
  'wind_speed',
  'wind_direction',
  'precipitation',
  'cloud_cover',
  'weather_code',
  'is_day',
] as const

const stationSeriesPath = (city: string, hours: number) =>
  `/weather/series?cities=${encodeURIComponent(city)}&hours=${hours}` +
  `&fields=${STATION_SERIES_FIELDS.join(',')}`

/**
 * One bulk observation request per station instead of `/weather/current/{city}`
 * + `/weather/history/{city}`: the newest point is the hero observation and the
 * ordered points are the chart series, so the two can never disagree about what
 * was observed.
 *
 * `step_hours` is left at the backend default (1 h = no downsampling), which is
 * what keeps the point count identical to the `/history` call it replaces.
 * Polling follows the old `useCurrentWeather` cadence (5 min): it was the faster
 * of the two removed queries, so the hero never gets staler than it used to.
 */
export const useStationSeries = (city: string, hours = 24) =>
  useApiQuery(queryKeys.series(city, hours), stationSeriesPath(city, hours), seriesResponseSchema, {
    refetchInterval: 5 * 60_000,
    enabled: Boolean(city),
  })

export const useStats = (city: string, hours = 24) =>
  useApiQuery(
    queryKeys.stats(city, hours),
    `/weather/stats/${encodeURIComponent(city)}?hours=${hours}`,
    statsResponseSchema,
    { enabled: Boolean(city), allowNotFound: true },
  )

/** Contract 2: M2 prediction horizons, mirrored by the backend config. */
export const PREDICTION_HORIZONS = [1, 3, 6, 12, 24] as const
export type PredictionHorizon = (typeof PREDICTION_HORIZONS)[number]

/**
 * Contract 2: prediction history for one city, unfiltered by horizon.
 *
 * One query per station: the page derives the latest run, the card row for the
 * selected horizon and the chart curve from the same window. Filtering by
 * horizon here would return at most one model point per inference run, i.e. a
 * point cloud instead of a forecast curve.
 */
export const usePredictions = (city: string, limit = 200) =>
  useApiQuery(
    queryKeys.predictions(city, limit),
    `/predictions/${encodeURIComponent(city)}?limit=${limit}`,
    z.array(predictionSchema),
    { enabled: Boolean(city), allowNotFound: true },
  )
