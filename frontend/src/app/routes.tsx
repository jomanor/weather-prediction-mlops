import { Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from '@/app/AppShell'
import { FEATURES } from '@/app/registry'
import { LoadingBlock } from '@/components/ui/Feedback'

/** Route table generated from the feature registry, one lazy chunk per route. */
export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        {FEATURES.flatMap((feature) =>
          feature.routes.map((route) => {
            const Page = route.component
            return (
              <Route
                key={`${feature.id}:${route.path || 'index'}`}
                index={route.path === ''}
                path={route.path || undefined}
                element={
                  <Suspense fallback={<LoadingBlock label="Cargando sección…" />}>
                    <Page />
                  </Suspense>
                }
              />
            )
          }),
        )}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
