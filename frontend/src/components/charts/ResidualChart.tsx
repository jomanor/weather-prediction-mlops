import { useMemo } from 'react'
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { SeriesPoint } from '@/api/schemas'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { usePreferences } from '@/app/preferences'
import { useChartSync } from '@/hooks/useChartSync'
import { formatDateTimeMs, formatSigned, isNum } from '@/lib/format'
import type { ChartPalette } from '@/lib/chart-theme'

interface ResidualChartProps {
  points: SeriesPoint[]
  height?: number
  palette: ChartPalette
}

/** Model error per hour, diverging around zero, on the shared time axis. */
export function ResidualChart({ points, height = 160, palette }: ResidualChartProps) {
  const { units } = usePreferences()
  const { domain } = useChartSync()

  const data = useMemo(
    () =>
      points
        .map((point) => ({
          ts: new Date(point.timestamp).getTime(),
          residual: isNum(point.residual_model)
            ? units === 'imperial'
              ? point.residual_model * 1.8
              : point.residual_model
            : null,
        }))
        .filter((row) => Number.isFinite(row.ts)),
    [points, units],
  )

  const bound = useMemo(() => {
    const values = data
      .map((row) => row.residual)
      .filter((value): value is number => value !== null)
      .map(Math.abs)
    return values.length ? Math.max(...values) * 1.15 : 1
  }, [data])

  const xDomain: [number, number] | ['dataMin', 'dataMax'] = domain ?? ['dataMin', 'dataMax']

  const axisProps = {
    stroke: palette.axis,
    tick: { fill: palette.textMuted, fontSize: 10, fontFamily: 'JetBrains Mono Variable, monospace' },
    tickLine: false,
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
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
          width={52}
          tickCount={5}
          tickFormatter={(value: number) => formatSigned(value, 1)}
          {...axisProps}
        />
        <Tooltip
          cursor={{ fill: palette.grid }}
          content={
            <ChartTooltip
              title={(label) => formatDateTimeMs(Number(label))}
              format={(entry) =>
                entry.value === null
                  ? '—'
                  : `${formatSigned(entry.value as number)} ${units === 'imperial' ? '°F' : '°C'}`
              }
            />
          }
        />
        <Bar
          dataKey="residual"
          name={`Error (observado − modelo, ${units === 'imperial' ? '°F' : '°C'})`}
          isAnimationActive={false}
        >
          {data.map((row, index) => (
            <Cell
              key={index}
              fill={row.residual !== null && row.residual >= 0 ? palette.bad : palette.accent}
              fillOpacity={0.75}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
