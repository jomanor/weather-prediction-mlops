import { useMemo } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { SeriesPoint } from '@/api/schemas'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { usePreferences } from '@/app/preferences'
import { convertTemperature, formatDateTime, formatNumber, isNum } from '@/lib/format'
import type { ChartPalette } from '@/lib/chart-theme'

interface BenchmarkChartProps {
  points: SeriesPoint[]
  height?: number
  palette: ChartPalette
}

/** Observed vs Spark GBT vs AEMET, on one temperature axis. */
export function BenchmarkChart({ points, height = 280, palette }: BenchmarkChartProps) {
  const { units } = usePreferences()

  const { data, domain } = useMemo(() => {
    const rows = points.map((point) => ({
      t: formatDateTime(point.timestamp),
      observed: isNum(point.observed) ? convertTemperature(point.observed, units) : null,
      model: isNum(point.model) ? convertTemperature(point.model, units) : null,
      aemet: isNum(point.aemet) ? convertTemperature(point.aemet, units) : null,
    }))

    const values = rows.flatMap((row) =>
      [row.observed, row.model, row.aemet].filter((value): value is number => value !== null),
    )
    if (!values.length) return { data: rows, domain: [0, 1] as [number, number] }

    const min = Math.min(...values)
    const max = Math.max(...values)
    const pad = Math.max((max - min) * 0.15, 1)
    return { data: rows, domain: [Math.floor(min - pad), Math.ceil(max + pad)] as [number, number] }
  }, [points, units])

  const axisProps = {
    stroke: palette.axis,
    tick: { fill: palette.textMuted, fontSize: 10, fontFamily: 'JetBrains Mono Variable, monospace' },
    tickLine: false,
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={palette.grid} vertical={false} />
        <XAxis dataKey="t" minTickGap={40} {...axisProps} />
        <YAxis
          domain={domain}
          width={44}
          tickFormatter={(value: number) => `${formatNumber(value, 0)}°`}
          {...axisProps}
        />
        <Tooltip
          content={
            <ChartTooltip
              format={(entry) => (entry.value === null ? '—' : `${formatNumber(entry.value as number)}°`)}
            />
          }
          cursor={{ stroke: palette.axis, strokeDasharray: '3 3' }}
        />
        <Line
          type="monotone"
          dataKey="observed"
          name="Observado"
          stroke={palette.observed}
          strokeWidth={2}
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="model"
          name="Spark GBT"
          stroke={palette.model}
          strokeWidth={1.75}
          strokeDasharray="6 3"
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="aemet"
          name="AEMET"
          stroke={palette.aemet}
          strokeWidth={1.5}
          strokeDasharray="2 3"
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
