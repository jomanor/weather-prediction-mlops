import { useEffect, useMemo, useState } from 'react'
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
import { buildHistoryChartData, type HistoryChartRow } from '@/components/charts/history-rows'
import { ChartBrush } from '@/components/charts/ChartBrush'
import { ChartTooltip, type TooltipEntry } from '@/components/charts/ChartTooltip'
import { SeriesLegend, type LegendItem } from '@/components/charts/SeriesLegend'
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
  formatDateTimeMs,
  formatDegrees,
  formatNumber,
  formatPercent,
  formatWithUnit,
  temperatureUnit,
  windUnit,
} from '@/lib/format'

export type HistoryVariable = 'temperature' | 'precipitation' | 'wind'

type SeriesKey = 'temperature' | 'model' | 'precipitation' | 'band' | 'wind'

interface HistoryChartProps {
  points: WeatherPoint[]
  /** Model predictions; temperature intervals draw the uncertainty band. */
  predictions?: Prediction[]
  height?: number
  palette: ChartPalette
  /** Selected observed variable; `temperature` also overlays precipitation. */
  variable?: HistoryVariable
  /** Nominal interval level (0–1), shown in the band legend entry. */
  intervalLevel?: number | null
}

interface ToggleState {
  variable: HistoryVariable
  hidden: SeriesKey[]
}

/** Observed history for the selected variable, on a shared time axis. */
export function HistoryChart({
  points,
  predictions = [],
  height = 260,
  palette,
  variable = 'temperature',
  intervalLevel = null,
}: HistoryChartProps) {
  const { units } = usePreferences()
  const { domain } = useChartSync()

  const {
    rows: data,
    temperatureDomain,
    precipitationMax,
    windDomain,
    hasInterval,
    hasModel,
  } = useMemo(() => buildHistoryChartData(points, predictions, units), [points, predictions, units])

  /* Series visibility is local to the chart: it resets when the variable
     changes (the series set is different) and survives range/brush changes. */
  const [toggles, setToggles] = useState<ToggleState>({ variable, hidden: [] })
  /* The series set changes with the variable, so visibility resets instead of
     carrying a hidden `temperature` key over to the precipitation chart. */
  useEffect(() => {
    setToggles((current) =>
      current.variable === variable ? current : { variable, hidden: [] },
    )
  }, [variable])
  const hidden = toggles.variable === variable ? toggles.hidden : []
  const isVisible = (key: SeriesKey) => !hidden.includes(key)

  const hasPrecipitation = useMemo(
    () => data.some((row) => row.precipitation !== null),
    [data],
  )

  const seriesKeys: SeriesKey[] =
    variable === 'temperature'
      ? [
          'temperature',
          ...(hasModel ? (['model'] as const) : []),
          ...(hasPrecipitation ? (['precipitation'] as const) : []),
          ...(hasInterval ? (['band'] as const) : []),
        ]
      : variable === 'precipitation'
        ? ['precipitation']
        : ['wind']

  const toggle = (rawId: string) => {
    const key = rawId as SeriesKey
    const current = toggles.variable === variable ? toggles.hidden : []
    const active = seriesKeys.filter((candidate) => !current.includes(candidate))
    /* Never hide the last visible series: an all-hidden plot is a dead end. */
    if (!current.includes(key) && active.length <= 1) return
    setToggles({
      variable,
      hidden: current.includes(key)
        ? current.filter((candidate) => candidate !== key)
        : [...current, key],
    })
  }

  const unitLabel = temperatureUnit(units)

  const legend: LegendItem[] = []
  if (variable === 'temperature') {
    legend.push({ id: 'temperature', label: 'Temperatura', color: palette.aemet })
    if (hasModel) legend.push({ id: 'model', label: 'Modelo', color: palette.model, dash: '4 3' })
    if (hasPrecipitation) {
      legend.push({ id: 'precipitation', label: 'Precipitación', color: palette.accent })
    }
    if (hasInterval) {
      const level =
        intervalLevel !== null ? ` ${formatPercent(intervalLevel * 100, 0)}` : ''
      legend.push({ id: 'band', label: `Banda${level}`, color: palette.model })
    }
  } else if (variable === 'precipitation') {
    legend.push({ id: 'precipitation', label: 'Precipitación', color: palette.accent })
  } else {
    legend.push({ id: 'wind', label: 'Viento', color: palette.observed })
  }
  const legendItems = legend.map((item) => ({
    ...item,
    active: isVisible((item.id ?? 'temperature') as SeriesKey),
  }))

  if (data.length === 0) {
    return (
      <div style={{ minHeight: height }} className="flex items-center justify-center">
        <EmptyState
          title="Sin observaciones en el rango"
          description="Amplía el rango temporal o comprueba que el backfill ha cargado datos históricos."
        />
      </div>
    )
  }

  /* The synced window ends at the last observation; extend it so the final
     forecast interval is not clipped off the right edge. */
  const lastTs = data[data.length - 1].ts
  const xDomain: [number, number] | ['dataMin', 'dataMax'] = domain
    ? [domain[0], Math.max(domain[1], lastTs)]
    : ['dataMin', 'dataMax']

  const axisProps = chartAxisProps(palette)

  const intervalRow = (entries: TooltipEntry[]) => {
    if (!hasInterval || !isVisible('band')) return null
    const row = entries[0]?.payload as HistoryChartRow | undefined
    if (!row || row.band_base === null || row.band_size === null) return null
    const upper = row.band_base + row.band_size
    return (
      <div className="flex items-center justify-between gap-3">
        <span className="text-[11px] text-fg-2">Banda</span>
        <span className="nums text-[11px] font-medium text-fg">
          {formatWithUnit(row.band_base, unitLabel, 1)} – {formatWithUnit(upper, unitLabel, 1)}
        </span>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={CHART_MARGIN}>
            <CartesianGrid {...chartGridProps(palette)} />
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
                  width={46}
                  tickFormatter={(value: number) => formatDegrees(value, 0)}
                  label={chartAxisLabel(palette, unitLabel, 'insideLeft')}
                  {...axisProps}
                />
                <YAxis
                  yAxisId="rain"
                  orientation="right"
                  domain={[0, precipitationMax]}
                  width={42}
                  tickFormatter={(value: number) => formatNumber(value, 0)}
                  label={chartAxisLabel(palette, 'mm', 'insideRight')}
                  {...axisProps}
                />
              </>
            ) : (
              <YAxis
                yAxisId="value"
                domain={variable === 'precipitation' ? [0, precipitationMax] : windDomain}
                width={46}
                tickFormatter={(value: number) =>
                  formatNumber(value, variable === 'precipitation' ? 1 : 0)
                }
                label={chartAxisLabel(
                  palette,
                  variable === 'precipitation' ? 'mm' : windUnit(units),
                  'insideLeft',
                )}
                {...axisProps}
              />
            )}

            <Tooltip
              cursor={chartLineCursor(palette)}
              content={
                <ChartTooltip
                  title={(label) => formatDateTimeMs(Number(label))}
                  format={(entry) => {
                    if (entry.value === null) return '—'
                    const value = entry.value as number
                    if (entry.dataKey === 'precipitation') {
                      return formatWithUnit(value, 'mm', 1)
                    }
                    if (entry.dataKey === 'wind') {
                      return formatWithUnit(value, windUnit(units), 1)
                    }
                    return formatWithUnit(value, unitLabel, 1)
                  }}
                  extra={intervalRow}
                />
              }
            />

            {variable === 'temperature' && hasInterval && isVisible('band') ? (
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

            {variable === 'temperature' && hasModel && isVisible('model') ? (
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

            {variable === 'temperature' && isVisible('precipitation') ? (
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

            {variable === 'precipitation' && isVisible('precipitation') ? (
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

            {variable === 'temperature' && isVisible('temperature') ? (
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

            {variable === 'wind' && isVisible('wind') ? (
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
      </div>

      <SeriesLegend
        items={legendItems}
        onToggle={toggle}
        label="Series del gráfico"
        className="px-1"
      />
    </div>
  )
}
