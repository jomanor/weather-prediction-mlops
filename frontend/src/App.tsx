import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from '@/app/AppShell'
import { BenchmarkPage } from '@/features/benchmark/BenchmarkPage'
import { ModelsPage } from '@/features/models/ModelsPage'
import { OverviewPage } from '@/features/overview/OverviewPage'
import { StationPage } from '@/features/station/StationPage'

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<OverviewPage />} />
        <Route path="stations" element={<StationPage />} />
        <Route path="benchmark" element={<BenchmarkPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
