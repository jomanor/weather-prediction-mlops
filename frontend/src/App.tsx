import { AppRoutes } from '@/app/routes'
import { ChartSyncProvider } from '@/hooks/useChartSync'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

export function App() {
  return (
    <StationSelectionProvider>
      <ChartSyncProvider>
        <AppRoutes />
      </ChartSyncProvider>
    </StationSelectionProvider>
  )
}
