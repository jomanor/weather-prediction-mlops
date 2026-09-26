import { Brush } from 'recharts'

import { brushDomain, type BrushRange } from '@/components/charts/chart-brush'
import type { ChartPalette } from '@/lib/chart-theme'
import { formatDateTimeMs } from '@/lib/format'
import { useChartSync } from '@/hooks/useChartSync'

interface ChartBrushProps {
  rows: ReadonlyArray<{ ts: number }>
  palette: ChartPalette
}

/**
 * U6: the brushing control that drives the shared time domain. It is remounted
 * whenever the page window changes (`resetToken`), so a window reset also
 * clears the brush selection instead of leaving stale indices behind.
 */
export function ChartBrush({ rows, palette }: ChartBrushProps) {
  const { resetToken, setDomain } = useChartSync()

  return (
    <Brush
      key={resetToken}
      dataKey="ts"
      height={20}
      travellerWidth={6}
      stroke={palette.axis}
      fill={palette.grid}
      ariaLabel="Seleccionar la ventana temporal"
      tickFormatter={(value) => formatDateTimeMs(Number(value))}
      onChange={(range: BrushRange) => setDomain(brushDomain(rows, range))}
    />
  )
}
