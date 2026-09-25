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

export const usePredictions = (city: string, limit = 48) =>
  useApiQuery(
    queryKeys.predictions(city, limit),
    `/predictions/${encodeURIComponent(city)}?limit=${limit}`,
    z.array(predictionSchema),
    { enabled: Boolean(city), allowNotFound: true },
  )
