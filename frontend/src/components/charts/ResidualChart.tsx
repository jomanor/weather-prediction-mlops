import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { SeriesPoint } from '@/api/schemas'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { usePreferences } from '@/app/preferences'
import { useChartSync } from '@/hooks/useChartSync'
import { EmptyState } from '@/components/ui/Feedback'
import {
  CHART_MARGIN,
  chartAxisLabel,
  chartAxisProps,
  chartBarCursor,
  chartGridProps,
  type ChartPalette,
} from '@/lib/chart-theme'
import {
  formatDateTimeMs,
  formatError,
  formatSigned,
  isNum,
  metricUnit,
} from '@/lib/format'

interface ResidualChartProps {
  points: SeriesPoint[]
  height?: number
  palette: ChartPalette
}

/**
 * Model error per hour, diverging around zero, on the shared time axis.
 *
 * One hue for both signs: the zero rule and the signed value carry direction,
 * so the chart never relies on a red/green pair to say which way the error went.
 */
export function ResidualChart({ points, height = 160, palette }: ResidualChartProps) {
  const { units } = usePreferences()
  const { domain } = useChartSync()

  const { data, bound, hasValues } = useMemo(() => {
    const rows = points
      .map((point) => ({
        ts: new Date(point.timestamp).getTime(),
        residual: isNum(point.residual_model)
          ? units === 'imperial'
            ? point.residual_model * 1.8
            : point.residual_model
          : null,
      }))
      .filter((row) => Number.isFinite(row.ts))

    const values = rows
      .map((row) => row.residual)
      .filter((value): value is number => value !== null)
      .map(Math.abs)
    return {
      data: rows,
      bound: values.length ? Math.max(...values) * 1.15 : 1,
      hasValues: values.length > 0,
    }
  }, [points, units])

  const unitLabel = metricUnit('temperature', units)
  const xDomain: [number, number] | ['dataMin', 'dataMax'] = domain ?? ['dataMin', 'dataMax']
  const axisProps = chartAxisProps(palette)

  if (!hasValues) {
    return (
      <div style={{ minHeight: height }} className="flex items-center justify-center">
        <EmptyState
          title="Sin residuos que mostrar"
          description="No hay pares observado/modelo en esta ventana temporal para calcular el error."
        />
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={CHART_MARGIN}>
        <CartesianGrid {...chartGridProps(palette)} />
        <ReferenceLine y={0} stroke={palette.axis} />
        <XAxis
          dataKey="ts"
          type="number"
          scale="time"
          domain={xDomain}
          minTickGap={48}
          tickFormatter={(value: number) => formatDateTimeMs(value)}
          {...axisProps}
        />
        <YAxis
          domain={[-bound, bound]}
          width={46}
          tickCount={5}
          tickFormatter={(value: number) => formatSigned(value, 1)}
          label={chartAxisLabel(palette, unitLabel, 'insideLeft')}
          {...axisProps}
        />
        <Tooltip
          cursor={chartBarCursor(palette)}
          content={
            <ChartTooltip
              title={(label) => formatDateTimeMs(Number(label))}
              format={(entry) =>
                entry.value === null ? '—' : formatError(entry.value as number, unitLabel)
              }
            />
          }
        />
        <Bar
          dataKey="residual"
          name={`Error (observado − modelo, ${unitLabel})`}
          fill={palette.model}
          fillOpacity={0.75}
          isAnimationActive={false}
          maxBarSize={14}
        />
      </BarChart>
    </ResponsiveContainer>
  )
}
