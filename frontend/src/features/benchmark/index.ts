import { Scale } from 'lucide-react'
import { lazy } from 'react'

import type { FeatureManifest } from '@/app/nav'

const BenchmarkPage = lazy(() =>
  import('@/features/benchmark/BenchmarkPage').then((module) => ({ default: module.BenchmarkPage })),
)

export const benchmarkFeature: FeatureManifest = {
  id: 'benchmark',
  navItem: { to: '/benchmark', label: 'Benchmark', icon: Scale, order: 30 },
  routes: [{ path: 'benchmark', component: BenchmarkPage }],
}
