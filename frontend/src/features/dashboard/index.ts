import { LayoutDashboard } from 'lucide-react'
import { lazy } from 'react'

import type { FeatureManifest } from '@/app/nav'

/**
 * Panel: the user-composable dashboard. Unlike the other three surfaces it has
 * no server-side content — the layout lives in `localStorage` and every card is
 * a read-only view of endpoints that already exist.
 */
const DashboardPage = lazy(() =>
  import('@/features/dashboard/DashboardPage').then((module) => ({
    default: module.DashboardPage,
  })),
)

export const dashboardFeature: FeatureManifest = {
  id: 'dashboard',
  navItem: { to: '/panel', label: 'Panel', icon: LayoutDashboard, order: 40 },
  routes: [{ path: 'panel', component: DashboardPage }],
}
