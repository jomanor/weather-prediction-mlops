import { useMutation, useQuery, useQueryClient, type UseQueryOptions } from '@tanstack/react-query'
import { z } from 'zod'

import { apiGet, apiSend } from './client'
import {
  benchmarkSchema,
  benchmarkSummarySchema,
  citySchema,
  currentWeatherSchema,
  geoSearchResponseSchema,
  healthSchema,
  historyResponseSchema,
  latestPredictionsSchema,
  modelsResponseSchema,
  predictionSchema,
  stationsResponseSchema,
  statsResponseSchema,
  type City,
} from './schemas'

const MINUTE = 60_000

export const queryKeys = {
  health: ['health'] as const,
  cities: ['cities'] as const,
  geo: (q: string) => ['geo', q] as const,
  stations: ['stations'] as const,
  station: (city: string) => ['station', city] as const,
  history: (city: string, hours: number) => ['history', city, hours] as const,
  stats: (city: string, hours: number) => ['stats', city, hours] as const,
  latestPredictions: ['predictions', 'latest'] as const,
  predictions: (city: string, limit: number) => ['predictions', city, limit] as const,
  benchmark: (city: string, hours: number) => ['benchmark', city, hours] as const,
  benchmarkSummary: ['benchmark', 'summary'] as const,
  models: ['models'] as const,
}

interface ApiQueryOptions<T> {
  enabled?: boolean
  refetchInterval?: number
  staleTime?: number
  allowNotFound?: boolean
  retry?: UseQueryOptions<T, Error, T, readonly unknown[]>['retry']
}

function useApiQuery<T>(
  key: readonly unknown[],
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  options: ApiQueryOptions<T> = {},
) {
  const { allowNotFound, ...rest } = options
  return useQuery<T, Error, T, readonly unknown[]>({
    queryKey: key,
    queryFn: ({ signal }) => apiGet(path, schema, { signal, allowNotFound }),
    staleTime: MINUTE,
    retry: 1,
    ...rest,
  })
}

export const useHealth = () =>
  useApiQuery(queryKeys.health, '/health', healthSchema, { refetchInterval: 5 * MINUTE })

export const useCities = () =>
  useApiQuery(queryKeys.cities, '/cities', z.array(citySchema), { staleTime: 30 * MINUTE })

/** Free-form location lookup against the Open-Meteo geocoding proxy. */
export function useGeoSearch(query: string) {
  return useApiQuery(queryKeys.geo(query), `/geo/search?q=${encodeURIComponent(query)}`, geoSearchResponseSchema, {
    enabled: query.trim().length >= 2,
    staleTime: 30 * MINUTE,
    retry: false,
  })
}

export function useAddCity() {
  const client = useQueryClient()
  return useMutation<City, Error, Omit<City, never>, unknown>({
    mutationFn: (city) =>
      apiSend('POST', '/cities', citySchema, {
        body: { name: city.name.trim(), latitude: city.latitude, longitude: city.longitude },
      }) as Promise<City>,
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: queryKeys.cities }),
        client.invalidateQueries({ queryKey: queryKeys.stations }),
      ])
    },
  })
}

/** Removes a station from the registry; historical data is kept upstream. */
export function useRemoveCity() {
  const client = useQueryClient()
  return useMutation<null, Error, string>({
    mutationFn: (name) =>
      apiSend('DELETE', `/cities/${encodeURIComponent(name)}`, null, { allowNotFound: true }),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: queryKeys.cities }),
        client.invalidateQueries({ queryKey: queryKeys.stations }),
      ])
    },
  })
}

export const useStations = () =>
  useApiQuery(queryKeys.stations, '/weather/current', stationsResponseSchema, {
    refetchInterval: 5 * MINUTE,
  })

export const useCurrentWeather = (city: string) =>
  useApiQuery(
    queryKeys.station(city),
    `/weather/current/${encodeURIComponent(city)}`,
    currentWeatherSchema,
    { refetchInterval: 5 * MINUTE, enabled: Boolean(city) },
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

export const useLatestPredictions = () =>
  useApiQuery(queryKeys.latestPredictions, '/predictions/latest', latestPredictionsSchema, {
    refetchInterval: 10 * MINUTE,
  })

export const usePredictions = (city: string, limit = 48) =>
  useApiQuery(
    queryKeys.predictions(city, limit),
    `/predictions/${encodeURIComponent(city)}?limit=${limit}`,
    z.array(predictionSchema),
    { enabled: Boolean(city), allowNotFound: true },
  )

export const useBenchmark = (city: string, hours = 24) =>
  useApiQuery(
    queryKeys.benchmark(city, hours),
    `/benchmark/${encodeURIComponent(city)}?hours=${hours}`,
    benchmarkSchema,
    { enabled: Boolean(city), refetchInterval: 30 * MINUTE },
  )

export const useBenchmarkSummary = () =>
  useApiQuery(queryKeys.benchmarkSummary, '/benchmark', benchmarkSummarySchema, {
    refetchInterval: 30 * MINUTE,
  })

export const useModels = () =>
  useApiQuery(queryKeys.models, '/models', modelsResponseSchema, { staleTime: 10 * MINUTE })
