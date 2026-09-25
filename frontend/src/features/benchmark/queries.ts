import { queryKeys, useApiQuery } from '@/api/queries'
import { benchmarkSchema } from '@/api/schemas'

/** Per-city observed/model/AEMET series and metrics. */
export const useBenchmark = (city: string, hours = 24) =>
  useApiQuery(
    queryKeys.benchmark(city, hours),
    `/benchmark/${encodeURIComponent(city)}?hours=${hours}`,
    benchmarkSchema,
    { enabled: Boolean(city), refetchInterval: 30 * 60_000 },
  )
