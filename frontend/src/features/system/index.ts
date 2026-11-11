import { Boxes } from 'lucide-react'
import { lazy } from 'react'

import type { FeatureManifest } from '@/app/nav'

/**
 * Sistema: the MLOps trust surface. Data quality, model registry and the
 * benchmark against AEMET used to be three top-level pages; they are three tabs
 * of one URL (`?tab=`) and the two old paths redirect here. Climate analytics
 * is no longer a tab: it lives in `ClimateViewsDialog` on the station page, so
 * `/analytics` redirects there.
 */
const SystemPage = lazy(() =>
  import('@/features/system/SystemPage').then((module) => ({ default: module.SystemPage })),
)

export const systemFeature: FeatureManifest = {
  id: 'system',
  navItem: { to: '/sistema', label: 'Sistema', icon: Boxes, order: 30 },
  routes: [{ path: 'sistema', component: SystemPage }],
  redirects: [
    { path: 'benchmark', to: '/sistema?tab=benchmark' },
    { path: 'models', to: '/sistema?tab=modelos' },
    { path: 'analytics', to: '/stations' },
  ],
}
