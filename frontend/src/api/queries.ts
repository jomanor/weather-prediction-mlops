import { useQuery, type UseQueryOptions } from '@tanstack/react-query'
import { z } from 'zod'

import { apiGet } from './client'
import { benchmarkSummarySchema, citySchema, healthSchema } from './schemas'

/**
 * Shared query layer.
 *
 * The contract lives in `client.ts`/`schemas.ts`; this module holds only the
 * cache-key map and the hooks that more than one feature needs. Feature-scoped
 * hooks live in `features/<name>/queries.ts` and build on `useApiQuery`.
 */

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

/** Shared `useQuery` wrapper: typed path, schema validation and sane defaults. */
export function useApiQuery<T>(
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

/** Station registry; used by the shell search, the overview and the charts. */
export const useCities = () =>
  useApiQuery(queryKeys.cities, '/cities', z.array(citySchema), { staleTime: 30 * MINUTE })

/** National MAE roll-up shared by the overview and the benchmark page. */
export const useBenchmarkSummary = () =>
  useApiQuery(queryKeys.benchmarkSummary, '/benchmark', benchmarkSummarySchema, {
    refetchInterval: 30 * MINUTE,
  })
