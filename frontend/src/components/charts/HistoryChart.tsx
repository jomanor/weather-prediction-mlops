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
import { useChartSync } from '@/hooks/useChartSync'
import { convertTemperature, formatDateTimeMs, formatNumber, isNum } from '@/lib/format'
import type { ChartPalette } from '@/lib/chart-theme'

export type HistoryVariable = 'temperature' | 'precipitation' | 'wind'

interface HistoryChartProps {
  points: WeatherPoint[]
  height?: number
  palette: ChartPalette
  /** Selected observed variable; `temperature` also overlays precipitation. */
  variable?: HistoryVariable
}

/** Observed history for the selected variable, on a shared time axis. */
export function HistoryChart({
  points,
  height = 260,
  palette,
  variable = 'temperature',
}: HistoryChartProps) {
  const { units } = usePreferences()
  const { domain } = useChartSync()

  const { data, temperatureDomain, precipitationMax, windDomain } = useMemo(() => {
    const rows = points
      .map((point) => ({
        ts: new Date(point.observed_at).getTime(),
        temperature: isNum(point.temperature) ? convertTemperature(point.temperature, units) : null,
        precipitation: point.precipitation,
        wind: isNum(point.wind_speed)
          ? units === 'imperial'
            ? point.wind_speed * 0.621371
            : point.wind_speed
          : null,
      }))
      .filter((row) => Number.isFinite(row.ts))

    const temperatures = rows
      .map((row) => row.temperature)
      .filter((value): value is number => value !== null)
    let temperatureDomain: [number, number] = [0, 1]
    if (temperatures.length) {
      const min = Math.min(...temperatures)
      const max = Math.max(...temperatures)
      const pad = Math.max((max - min) * 0.15, 1)
      temperatureDomain = [Math.floor(min - pad), Math.ceil(max + pad)]
    }

    const precipitationMax = Math.max(1, ...rows.map((row) => row.precipitation ?? 0)) * 1.2

    const winds = rows.map((row) => row.wind).filter((value): value is number => value !== null)
    let windDomain: [number, number] = [0, 1]
    if (winds.length) {
      const min = Math.min(...winds)
      const max = Math.max(...winds)
      const pad = Math.max((max - min) * 0.15, 1)
      windDomain = [Math.max(0, Math.floor(min - pad)), Math.ceil(max + pad)]
    }

    return { data: rows, temperatureDomain, precipitationMax, windDomain }
  }, [points, units])

  const xDomain: [number, number] | ['dataMin', 'dataMax'] = domain ?? ['dataMin', 'dataMax']

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
      </ComposedChart>
    </ResponsiveContainer>
  )
}
