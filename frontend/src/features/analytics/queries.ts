import { useApiQuery } from '@/api/queries'
import {
  ANALYTICS_BOUNDS,
  clampDays,
} from '@/features/analytics/transforms'
import { diurnalSchema, windRoseSchema } from '@/features/analytics/schemas'

/**
 * Analytics query hooks (Contract 4). Every `days` parameter is clamped to the
 * endpoint's validated bound before it reaches the URL; results are cached for
 * 5 minutes, matching the backend's `cached()` TTL.
 */

const STALE = 5 * 60_000

/* `daily` is gone with the Analítica tab (audit 16); the heatmap and the wind
   rose are the two retained views, and each keeps its own cache key. */
export const analyticsKeys = {
  windRose: (city: string, days: number) => ['analytics', 'wind-rose', city, days] as const,
  diurnal: (days: number) => ['analytics', 'diurnal', days] as const,
}

export function useAnalyticsWindRose(city: string, days: number = ANALYTICS_BOUNDS.windRose.default) {
  const bounded = clampDays(days, ANALYTICS_BOUNDS.windRose)
  return useApiQuery(
    analyticsKeys.windRose(city, bounded),
    `/analytics/wind-rose?city=${encodeURIComponent(city)}&days=${bounded}`,
    windRoseSchema,
    { enabled: Boolean(city), staleTime: STALE },
  )
}

export function useAnalyticsDiurnal(days: number = ANALYTICS_BOUNDS.diurnal.default) {
  const bounded = clampDays(days, ANALYTICS_BOUNDS.diurnal)
  return useApiQuery(
    analyticsKeys.diurnal(bounded),
    `/analytics/diurnal?days=${bounded}`,
    diurnalSchema,
    { staleTime: STALE },
  )
}
