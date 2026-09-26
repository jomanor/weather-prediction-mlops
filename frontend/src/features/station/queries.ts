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

export const usePredictions = (city: string, limit = 48, horizon: number = 1) =>
  useApiQuery(
    queryKeys.predictions(city, limit, horizon),
    `/predictions/${encodeURIComponent(city)}?limit=${limit}&horizon=${horizon}`,
    z.array(predictionSchema),
    { enabled: Boolean(city), allowNotFound: true },
  )
