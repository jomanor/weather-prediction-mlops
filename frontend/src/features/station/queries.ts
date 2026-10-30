import { z } from 'zod'

import { queryKeys, useApiQuery } from '@/api/queries'
import {
  currentWeatherSchema,
  historyResponseSchema,
  predictionSchema,
  statsResponseSchema,
} from '@/api/schemas'

/** Station-scoped queries: current observation, history, stats and predictions. */

export const useCurrentWeather = (city: string) =>
  useApiQuery(
    queryKeys.station(city),
    `/weather/current/${encodeURIComponent(city)}`,
    currentWeatherSchema,
    { refetchInterval: 5 * 60_000, enabled: Boolean(city) },
  )

export const useHistory = (city: string, hours = 24) =>
  useApiQuery(
    queryKeys.history(city, hours),
    `/weather/history/${encodeURIComponent(city)}?hours=${hours}&limit=500`,
    historyResponseSchema,
    { enabled: Boolean(city) },
  )

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
