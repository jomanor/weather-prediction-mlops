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
import { ChartBrush } from '@/components/charts/ChartBrush'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { usePreferences } from '@/app/preferences'
import { useChartSync } from '@/hooks/useChartSync'
import { EmptyState } from '@/components/ui/Feedback'
import {
  CHART_MARGIN,
  chartAxisLabel,
  chartAxisProps,
  chartGridProps,
  chartLineCursor,
  type ChartPalette,
} from '@/lib/chart-theme'
import {
  convertTemperature,
  formatDateTimeMs,
  formatDegrees,
  formatWithUnit,
  isNum,
  temperatureUnit,
} from '@/lib/format'

interface BenchmarkChartProps {
  points: SeriesPoint[]
  height?: number
  palette: ChartPalette
}

/** Observed vs Spark GBT vs AEMET, on one temperature axis. */
export function BenchmarkChart({ points, height = 280, palette }: BenchmarkChartProps) {
  const { units } = usePreferences()
  const { domain } = useChartSync()

  const { data, valueDomain, hasValues } = useMemo(() => {
    const rows = points
      .map((point) => ({
        ts: new Date(point.timestamp).getTime(),
        observed: isNum(point.observed) ? convertTemperature(point.observed, units) : null,
        model: isNum(point.model) ? convertTemperature(point.model, units) : null,
        aemet: isNum(point.aemet) ? convertTemperature(point.aemet, units) : null,
      }))
      .filter((row) => Number.isFinite(row.ts))

    const values = rows.flatMap((row) =>
      [row.observed, row.model, row.aemet].filter((value): value is number => value !== null),
    )
    if (!values.length) return { data: rows, valueDomain: [0, 1] as [number, number], hasValues: false }

    const min = Math.min(...values)
    const max = Math.max(...values)
    const pad = Math.max((max - min) * 0.15, 1)
    return {
      data: rows,
      valueDomain: [Math.floor(min - pad), Math.ceil(max + pad)] as [number, number],
      hasValues: true,
    }
  }, [points, units])

  const unitLabel = temperatureUnit(units)
  const xDomain: [number, number] | ['dataMin', 'dataMax'] = domain ?? ['dataMin', 'dataMax']
  const axisProps = chartAxisProps(palette)

  if (!hasValues) {
    return (
      <div style={{ minHeight: height }} className="flex items-center justify-center">
        <EmptyState
          title="Sin datos comparables"
          description="No hay observaciones ni predicciones con temperatura en esta ventana temporal."
        />
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={CHART_MARGIN}>
        <CartesianGrid {...chartGridProps(palette)} />
        <XAxis
          dataKey="ts"
          type="number"
          scale="time"
          domain={xDomain}
          minTickGap={40}
          tickFormatter={(value: number) => formatDateTimeMs(value)}
          {...axisProps}
        />
        <YAxis
          domain={valueDomain}
          width={46}
          tickFormatter={(value: number) => formatDegrees(value, 0)}
          label={chartAxisLabel(palette, unitLabel, 'insideLeft')}
          {...axisProps}
        />
        <Tooltip
          content={
            <ChartTooltip
              title={(label) => formatDateTimeMs(Number(label))}
              format={(entry) =>
                entry.value === null
                  ? '—'
                  : formatWithUnit(entry.value as number, unitLabel, 1)
              }
            />
          }
          cursor={chartLineCursor(palette)}
        />
        <Line
          type="monotone"
          dataKey="observed"
          name="Observado"
          stroke={palette.observed}
          strokeWidth={1.5}
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="model"
          name="Spark GBT"
          stroke={palette.model}
          strokeWidth={1.25}
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
        <ChartBrush rows={data} palette={palette} />
      </LineChart>
    </ResponsiveContainer>
  )
}
