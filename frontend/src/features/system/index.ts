import { Boxes } from 'lucide-react'
import { lazy } from 'react'

import type { FeatureManifest } from '@/app/nav'

/**
 * Sistema: the MLOps trust surface. Data quality, model registry, benchmark
 * against AEMET and climate analytics used to be four top-level pages; they are
 * four tabs of one URL (`?tab=`) and the three old paths redirect here.
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
    { path: 'analytics', to: '/sistema?tab=analitica' },
  ],
}
