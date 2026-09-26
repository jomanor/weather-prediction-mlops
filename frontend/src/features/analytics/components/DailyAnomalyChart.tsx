import { useMemo } from 'react'
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { heatwaveRuns } from '@/features/analytics/transforms'
import type { DailyPoint } from '@/features/analytics/schemas'
import type { ChartPalette } from '@/lib/chart-theme'
import { formatDate, formatNumber, formatSigned } from '@/lib/format'

/**
 * Daily anomaly with heating/cooling degree days (base 18 °C). Bars stack HDD
 * and CDD on the left axis; the anomaly line sits on the right axis. Heatwave
 * days (≥3 consecutive days with tmax ≥ 35 °C) are banded behind the series so
 * the marker is legible even where the bars are tall.
 */
export function DailyAnomalyChart({
  points,
  palette,
  height = 260,
}: {
  points: readonly DailyPoint[]
  palette: ChartPalette
  height?: number
}) {
  const runs = useMemo(() => heatwaveRuns(points), [points])

  const axisProps = {
    stroke: palette.axis,
    tick: {
      fill: palette.textMuted,
      fontSize: 10,
      fontFamily: 'JetBrains Mono Variable, monospace',
    },
    tickLine: false,
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={points as DailyPoint[]} margin={{ top: 8, right: 4, bottom: 0, left: -14 }}>
        <CartesianGrid stroke={palette.grid} strokeDasharray="2 4" vertical={false} />

        {runs.map((run) => (
          <ReferenceArea
            key={run.start}
            x1={points[run.start].date}
            x2={points[run.end].date}
            yAxisId="anomaly"
            fill={palette.bad}
            fillOpacity={0.12}
            stroke={palette.bad}
            strokeOpacity={0.35}
            strokeDasharray="2 3"
          />
        ))}

        <XAxis
          dataKey="date"
          minTickGap={32}
          tickFormatter={(value: string) => value.slice(5).replace('-', '/')}
          {...axisProps}
        />
        <YAxis
          yAxisId="degree"
          width={30}
          tickFormatter={(value: number) => formatNumber(value, 0)}
          {...axisProps}
        />
        <YAxis
          yAxisId="anomaly"
          orientation="right"
          width={40}
          tickFormatter={(value: number) => `${formatSigned(value, 0)}°`}
          {...axisProps}
        />

        <Tooltip
          cursor={{ stroke: palette.axis, strokeDasharray: '3 3' }}
          content={
            <ChartTooltip
              title={(label) => formatDate(String(label))}
              format={(entry) => {
                if (typeof entry.value !== 'number' || !Number.isFinite(entry.value)) return '—'
                if (entry.dataKey === 'anomaly') return `${formatSigned(entry.value, 1)} °C`
                return `${formatNumber(entry.value, 1)} °C·d`
              }}
            />
          }
        />

        <Bar
          yAxisId="degree"
          dataKey="hdd"
          name="Grados de calefacción"
          stackId="degree"
          fill={palette.accent}
          fillOpacity={0.7}
          isAnimationActive={false}
          maxBarSize={18}
        />
        <Bar
          yAxisId="degree"
          dataKey="cdd"
          name="Grados de refrigeración"
          stackId="degree"
          fill={palette.aemet}
          fillOpacity={0.75}
          isAnimationActive={false}
          maxBarSize={18}
        />
        <Line
          yAxisId="anomaly"
          dataKey="anomaly"
          name="Anomalía"
          type="monotone"
          stroke={palette.model}
          strokeWidth={1.5}
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
