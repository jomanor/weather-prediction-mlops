import { useApiQuery } from '@/api/queries'
import {
  ANALYTICS_BOUNDS,
  clampDays,
} from '@/features/analytics/transforms'
import {
  analyticsDailySchema,
  climatologySchema,
  correlationSchema,
  diurnalSchema,
  errorByHourSchema,
  windRoseSchema,
  type AnalyticsVariable,
} from '@/features/analytics/schemas'

/**
 * Analytics query hooks (Contract 4). Every `days` parameter is clamped to the
 * endpoint's validated bound before it reaches the URL; results are cached for
 * 5 minutes, matching the backend's `cached()` TTL.
 */

const STALE = 5 * 60_000

export const analyticsKeys = {
  daily: (city: string, days: number) => ['analytics', 'daily', city, days] as const,
  climatology: (city: string) => ['analytics', 'climatology', city] as const,
  windRose: (city: string, days: number) => ['analytics', 'wind-rose', city, days] as const,
  diurnal: (days: number) => ['analytics', 'diurnal', days] as const,
  correlation: (days: number, variable: AnalyticsVariable) =>
    ['analytics', 'correlation', days, variable] as const,
  errorByHour: (days: number) => ['analytics', 'error-by-hour', days] as const,
}

export function useAnalyticsDaily(city: string, days: number = ANALYTICS_BOUNDS.daily.default) {
  const bounded = clampDays(days, ANALYTICS_BOUNDS.daily)
  return useApiQuery(
    analyticsKeys.daily(city, bounded),
    `/analytics/daily?city=${encodeURIComponent(city)}&days=${bounded}`,
    analyticsDailySchema,
    { enabled: Boolean(city), staleTime: STALE },
  )
}

export function useAnalyticsClimatology(city: string) {
  return useApiQuery(
    analyticsKeys.climatology(city),
    `/analytics/climatology?city=${encodeURIComponent(city)}`,
    climatologySchema,
    { enabled: Boolean(city), staleTime: 30 * 60_000 },
  )
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

export function useAnalyticsCorrelation(
  days: number = ANALYTICS_BOUNDS.correlation.default,
  variable: AnalyticsVariable = 'temperature',
) {
  const bounded = clampDays(days, ANALYTICS_BOUNDS.correlation)
  return useApiQuery(
    analyticsKeys.correlation(bounded, variable),
    `/analytics/correlation?days=${bounded}&var=${variable}`,
    correlationSchema,
    { staleTime: 30 * 60_000 },
  )
}

export function useAnalyticsErrorByHour(days: number = ANALYTICS_BOUNDS.errorByHour.default) {
  const bounded = clampDays(days, ANALYTICS_BOUNDS.errorByHour)
  return useApiQuery(
    analyticsKeys.errorByHour(bounded),
    `/analytics/error-by-hour?days=${bounded}`,
    errorByHourSchema,
    { staleTime: STALE },
  )
}
