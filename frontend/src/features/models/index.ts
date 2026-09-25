import { Boxes } from 'lucide-react'
import { lazy } from 'react'

import type { FeatureManifest } from '@/app/nav'

const ModelsPage = lazy(() =>
  import('@/features/models/ModelsPage').then((module) => ({ default: module.ModelsPage })),
)

export const modelsFeature: FeatureManifest = {
  id: 'models',
  navItem: { to: '/models', label: 'Modelos', icon: Boxes, order: 40 },
  routes: [{ path: 'models', component: ModelsPage }],
}
