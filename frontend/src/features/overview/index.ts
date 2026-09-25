import { LayoutGrid } from 'lucide-react'
import { lazy } from 'react'

import type { FeatureManifest } from '@/app/nav'

const OverviewPage = lazy(() =>
  import('@/features/overview/OverviewPage').then((module) => ({ default: module.OverviewPage })),
)

export const overviewFeature: FeatureManifest = {
  id: 'overview',
  navItem: { to: '/', label: 'Resumen', icon: LayoutGrid, order: 10 },
  routes: [{ path: '', component: OverviewPage }],
}
