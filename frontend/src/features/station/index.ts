import { Thermometer } from 'lucide-react'
import { lazy } from 'react'

import type { FeatureManifest } from '@/app/nav'

const StationPage = lazy(() =>
  import('@/features/station/StationPage').then((module) => ({ default: module.StationPage })),
)

export const stationFeature: FeatureManifest = {
  id: 'station',
  navItem: { to: '/stations', label: 'Estación', icon: Thermometer, order: 20 },
  routes: [{ path: 'stations', component: StationPage }],
}
