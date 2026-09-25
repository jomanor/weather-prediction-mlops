import { useMemo } from 'react'
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { Prediction, WeatherPoint } from '@/api/schemas'
import { buildHistoryChartData } from '@/components/charts/history-rows'
import { ChartBrush } from '@/components/charts/ChartBrush'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { usePreferences } from '@/app/preferences'
import { useChartSync } from '@/hooks/useChartSync'
import { formatDateTimeMs, formatNumber } from '@/lib/format'
import type { ChartPalette } from '@/lib/chart-theme'

export type HistoryVariable = 'temperature' | 'precipitation' | 'wind'

interface HistoryChartProps {
  points: WeatherPoint[]
  /** Model predictions; temperature intervals draw the uncertainty band. */
  predictions?: Prediction[]
  height?: number
  palette: ChartPalette
  /** Selected observed variable; `temperature` also overlays precipitation. */
  variable?: HistoryVariable
}

/** Observed history for the selected variable, on a shared time axis. */
export function HistoryChart({
  points,
  predictions = [],
  height = 260,
  palette,
  variable = 'temperature',
}: HistoryChartProps) {
  const { units } = usePreferences()
  const { domain } = useChartSync()

  const { rows: data, temperatureDomain, precipitationMax, windDomain, hasInterval, hasModel } =
    useMemo(() => buildHistoryChartData(points, predictions, units), [points, predictions, units])

  /* The synced window ends at the last observation; extend it so the final
     forecast interval is not clipped off the right edge. */
  const lastTs = data.length ? data[data.length - 1].ts : 0
  const xDomain: [number, number] | ['dataMin', 'dataMax'] = domain
    ? [domain[0], Math.max(domain[1], lastTs)]
    : ['dataMin', 'dataMax']

  const axisProps = {
    stroke: palette.axis,
    tick: { fill: palette.textMuted, fontSize: 10, fontFamily: 'JetBrains Mono Variable, monospace' },
    tickLine: false,
  }

  const unitLabel = units === 'imperial' ? '°F' : '°C'

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={palette.grid} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="ts"
          type="number"
          scale="time"
          domain={xDomain}
          minTickGap={48}
          tickFormatter={(value: number) => formatDateTimeMs(value)}
          {...axisProps}
        />

        {variable === 'temperature' ? (
          <>
            <YAxis
              yAxisId="value"
              domain={temperatureDomain}
              width={44}
              tickFormatter={(value: number) => `${formatNumber(value, 0)}°`}
              {...axisProps}
            />
            <YAxis
              yAxisId="rain"
              orientation="right"
              domain={[0, precipitationMax]}
              width={30}
              tickFormatter={(value: number) => formatNumber(value, 0)}
              {...axisProps}
            />
          </>
        ) : (
          <YAxis
            yAxisId="value"
            domain={variable === 'precipitation' ? [0, precipitationMax] : windDomain}
            width={44}
            tickFormatter={(value: number) => formatNumber(value, variable === 'precipitation' ? 1 : 0)}
            {...axisProps}
          />
        )}

        <Tooltip
          cursor={{ stroke: palette.axis, strokeDasharray: '3 3' }}
          content={
            <ChartTooltip
              title={(label) => formatDateTimeMs(Number(label))}
              format={(entry) => {
                if (entry.value === null) return '—'
                if (entry.dataKey === 'precipitation') return `${formatNumber(entry.value as number, 1)} mm`
                if (entry.dataKey === 'wind') {
                  return `${formatNumber(entry.value as number, 1)} ${units === 'imperial' ? 'mph' : 'km/h'}`
                }
                return `${formatNumber(entry.value as number)}${unitLabel}`
              }}
            />
          }
        />

        {variable === 'temperature' && hasInterval ? (
          <>
            {/* Range band as two stacked areas: transparent base lifts the
                tinted size area to [lower, upper]. */}
            <Area
              yAxisId="value"
              type="linear"
              dataKey="band_base"
              stackId="interval"
              stroke="none"
              fill="transparent"
              connectNulls={false}
              isAnimationActive={false}
              tooltipType="none"
            />
            <Area
              yAxisId="value"
              type="linear"
              dataKey="band_size"
              stackId="interval"
              name="Banda"
              stroke="none"
              fill={palette.model}
              fillOpacity={0.16}
              connectNulls={false}
              isAnimationActive={false}
              tooltipType="none"
            />
          </>
        ) : null}

        {variable === 'temperature' && hasModel ? (
          <Line
            yAxisId="value"
            type="monotone"
            dataKey="model"
            name="Modelo"
            stroke={palette.model}
            strokeWidth={1.5}
            strokeDasharray="4 3"
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        ) : null}

        {variable === 'temperature' ? (
          <Bar
            yAxisId="rain"
            dataKey="precipitation"
            name="Precipitación"
            fill={palette.accent}
            fillOpacity={0.45}
            isAnimationActive={false}
            maxBarSize={14}
          />
        ) : null}

        {variable === 'precipitation' ? (
          <Bar
            yAxisId="value"
            dataKey="precipitation"
            name="Precipitación"
            fill={palette.accent}
            fillOpacity={0.6}
            isAnimationActive={false}
            maxBarSize={14}
          />
        ) : null}

        {variable === 'temperature' ? (
          <Line
            yAxisId="value"
            type="monotone"
            dataKey="temperature"
            name="Temperatura"
            stroke={palette.aemet}
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        ) : null}

        {variable === 'wind' ? (
          <Line
            yAxisId="value"
            type="monotone"
            dataKey="wind"
            name="Viento"
            stroke={palette.observed}
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        ) : null}

        <ChartBrush rows={data} palette={palette} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
