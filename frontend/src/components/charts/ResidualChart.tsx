import { useMemo } from 'react'
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { SeriesPoint } from '@/api/schemas'
import { ChartTooltip } from '@/components/charts/ChartTooltip'
import { usePreferences } from '@/app/preferences'
import { formatDateTime, formatSigned, isNum } from '@/lib/format'
import type { ChartPalette } from '@/lib/chart-theme'

interface ResidualChartProps {
  points: SeriesPoint[]
  height?: number
  palette: ChartPalette
}

/** Model error per hour, diverging around zero. */
export function ResidualChart({ points, height = 160, palette }: ResidualChartProps) {
  const { units } = usePreferences()

  const data = useMemo(
    () =>
      points.map((point) => ({
        t: formatDateTime(point.timestamp),
        residual: isNum(point.residual_model)
          ? units === 'imperial'
            ? point.residual_model * 1.8
            : point.residual_model
          : null,
      })),
    [points, units],
  )

  const bound = useMemo(() => {
    const values = data
      .map((row) => row.residual)
      .filter((value): value is number => value !== null)
      .map(Math.abs)
    return values.length ? Math.max(...values) * 1.15 : 1
  }, [data])

  const axisProps = {
    stroke: palette.axis,
    tick: { fill: palette.textMuted, fontSize: 10, fontFamily: 'JetBrains Mono Variable, monospace' },
    tickLine: false,
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <ReferenceLine y={0} stroke={palette.axis} />
        <XAxis dataKey="t" minTickGap={48} {...axisProps} />
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
