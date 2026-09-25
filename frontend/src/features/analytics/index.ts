import { ChartLine } from 'lucide-react'
import { lazy } from 'react'

import type { FeatureManifest } from '@/app/nav'

/**
 * Analytics feature (Batch 4, Contract 4, WS-D2).
 *
 * Renders the diurnal heatmap, the daily anomaly / HDD-CDD view and the wind
 * rose from the `/api/analytics/*` endpoints. `app/registry.ts` owns the import
 * that mounts it (WS-D1), so this folder stays self-contained.
 */

const AnalyticsPage = lazy(() =>
  import('@/features/analytics/AnalyticsPage').then((module) => ({
    default: module.AnalyticsPage,
  })),
)

export const analyticsFeature: FeatureManifest = {
  id: 'analytics',
  navItem: { to: '/analytics', label: 'Analítica', icon: ChartLine, order: 50 },
  routes: [{ path: 'analytics', component: AnalyticsPage }],
}
