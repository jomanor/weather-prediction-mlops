import { useMemo } from 'react'
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { WeatherPoint } from '@/api/schemas'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { usePreferences } from '@/app/preferences'
import { convertTemperature, formatDateTime, formatNumber, isNum } from '@/lib/format'
import type { ChartPalette } from '@/lib/chart-theme'

interface HistoryChartProps {
  points: WeatherPoint[]
  height?: number
  palette: ChartPalette
}

/** Observed temperature over precipitation, the two variables that drive the model. */
export function HistoryChart({ points, height = 260, palette }: HistoryChartProps) {
  const { units } = usePreferences()

  const { data, domain } = useMemo(() => {
    const rows = points.map((point) => ({
      t: formatDateTime(point.observed_at),
      temperature: isNum(point.temperature) ? convertTemperature(point.temperature, units) : null,
      precipitation: point.precipitation,
    }))

    const temperatures = rows
      .map((row) => row.temperature)
      .filter((value): value is number => value !== null)

    if (!temperatures.length) return { data: rows, domain: [0, 1] as [number, number] }
    const min = Math.min(...temperatures)
    const max = Math.max(...temperatures)
    const pad = Math.max((max - min) * 0.15, 1)
    return { data: rows, domain: [Math.floor(min - pad), Math.ceil(max + pad)] as [number, number] }
  }, [points, units])

  const maxPrecipitation = useMemo(
    () => Math.max(1, ...data.map((row) => row.precipitation ?? 0)) * 1.2,
    [data],
  )

  const axisProps = {
    stroke: palette.axis,
    tick: { fill: palette.textMuted, fontSize: 10, fontFamily: 'JetBrains Mono Variable, monospace' },
    tickLine: false,
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={palette.grid} vertical={false} />
        <XAxis dataKey="t" minTickGap={48} {...axisProps} />
        <YAxis
          yAxisId="temperature"
          domain={domain}
          width={44}
          tickFormatter={(value: number) => `${formatNumber(value, 0)}°`}
          {...axisProps}
        />
        <YAxis
          yAxisId="precipitation"
          orientation="right"
          domain={[0, maxPrecipitation]}
          width={30}
          tickFormatter={(value: number) => formatNumber(value, 0)}
          {...axisProps}
        />
        <Tooltip
          cursor={{ stroke: palette.axis, strokeDasharray: '3 3' }}
          content={
            <ChartTooltip
              format={(entry) =>
                entry.dataKey === 'precipitation'
                  ? `${formatNumber(entry.value as number, 1)} mm`
                  : `${formatNumber(entry.value as number)}°`
              }
            />
          }
        />
        <Bar
          yAxisId="precipitation"
          dataKey="precipitation"
          name="Precipitación"
          fill={palette.accent}
          fillOpacity={0.45}
          isAnimationActive={false}
          maxBarSize={14}
        />
        <Line
          yAxisId="temperature"
          type="monotone"
          dataKey="temperature"
          name="Temperatura"
          stroke={palette.aemet}
          strokeWidth={2}
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
