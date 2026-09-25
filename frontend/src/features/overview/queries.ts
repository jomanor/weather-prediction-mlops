import { useMutation, useQueryClient } from '@tanstack/react-query'

import { apiSend } from '@/api/client'
import { queryKeys, useApiQuery } from '@/api/queries'
import {
  citySchema,
  geoSearchResponseSchema,
  latestPredictionsSchema,
  stationsResponseSchema,
  type City,
} from '@/api/schemas'

/** Overview queries: national station list, latest predictions and city registry. */

export const useStations = () =>
  useApiQuery(queryKeys.stations, '/weather/current', stationsResponseSchema, {
    refetchInterval: 5 * 60_000,
  })

export const useLatestPredictions = () =>
  useApiQuery(queryKeys.latestPredictions, '/predictions/latest', latestPredictionsSchema, {
    refetchInterval: 10 * 60_000,
  })

/** Free-form location lookup against the Open-Meteo geocoding proxy. */
export function useGeoSearch(query: string) {
  return useApiQuery(queryKeys.geo(query), `/geo/search?q=${encodeURIComponent(query)}`, geoSearchResponseSchema, {
    enabled: query.trim().length >= 2,
    staleTime: 30 * 60_000,
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
