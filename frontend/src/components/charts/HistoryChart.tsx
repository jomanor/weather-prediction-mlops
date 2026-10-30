import { Minus, Plus } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
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
import { unionDomain } from '@/components/charts/chart-zoom'
import { buildHistoryChartData, type HistoryChartRow } from '@/components/charts/history-rows'
import { ChartTooltip, type TooltipEntry } from '@/components/charts/ChartTooltip'
import { SeriesLegend, type LegendItem } from '@/components/charts/SeriesLegend'
import { useChartZoom } from '@/components/charts/useChartZoom'
import { usePreferences } from '@/app/preferences'
import { useChartSync, type TimeDomain } from '@/hooks/useChartSync'
import { Button } from '@/components/ui/Button'
import { EmptyState } from '@/components/ui/Feedback'
import {
  CHART_MARGIN,
  SERIES_DASH,
  chartAxisLabel,
  chartAxisProps,
  chartGridProps,
  chartLineCursor,
  type ChartPalette,
} from '@/lib/chart-theme'
import {
  formatDateTimeMs,
  formatMetric,
  formatMetricTick,
  formatPercent,
  metricForSeries,
  metricUnit,
} from '@/lib/format'

export type HistoryVariable = 'temperature' | 'precipitation' | 'wind'

type SeriesKey = 'temperature' | 'model' | 'precipitation' | 'band' | 'wind'

/* Y-axis gutters. The same widths feed the pointer→time mapping, so a wheel
   notch zooms around the cursor instead of around the axis edge. */
const AXIS_LEFT = 46
const AXIS_RIGHT_RAIN = 42

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
  const { domain, resetToken, setDomain } = useChartSync()

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

  /* Zoom state. `bounds` is the widest reachable window — the page window
     (which can end before the last forecast row) unioned with the data extent —
     latched when the shared window resets so zooming never shrinks the range
     the user can zoom back out to. `zoomDomain` is the user's selection; it is
     local because the shared domain is also written by the page window. */
  const dataBounds = useMemo<TimeDomain | null>(
    () => (data.length > 0 ? [data[0].ts, data[data.length - 1].ts] : null),
    [data],
  )
  const [base, setBase] = useState<TimeDomain | null>(null)
  const [zoomDomain, setZoomDomain] = useState<TimeDomain | null>(null)

  const dataStart = dataBounds ? dataBounds[0] : null
  const dataEnd = dataBounds ? dataBounds[1] : null
  useEffect(() => {
    /* `domain` is read when the window resets, not tracked: a zoom writes the
       shared domain too, and re-latching on it would defeat the zoom. */
    setBase(unionDomain(domain, dataBounds))
    setZoomDomain(null)
  }, [resetToken, dataStart, dataEnd])

  /* Union with the data extent covers the forecast rows that run past the
     page window, so the last interval is never clipped; before the latch
     effect runs, the same expression is the correct initial view. */
  const bounds = base ?? unionDomain(domain, dataBounds)
  const view = zoomDomain ?? bounds

  const insets = useMemo(
    () => ({
      left: AXIS_LEFT + CHART_MARGIN.left,
      right: (variable === 'temperature' ? AXIS_RIGHT_RAIN : 0) + CHART_MARGIN.right,
    }),
    [variable],
  )

  /* The chart owns the view and publishes it, so every chart on the page that
     reads the shared domain (benchmark, residuals) follows the same window. */
  const publishZoom = (next: TimeDomain | null) => {
    setZoomDomain(next)
    setDomain(next)
  }
  const zoom = useChartZoom({ view, bounds, onChange: publishZoom, insets })

  /* A different series set is a different view: drop the zoom with it. */
  const previousVariable = useRef(variable)
  useEffect(() => {
    if (previousVariable.current === variable) return
    previousVariable.current = variable
    publishZoom(null)
  }, [variable])

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

  const unitLabel = metricUnit('temperature', units)

  const legend: LegendItem[] = []
  if (variable === 'temperature') {
    legend.push({ id: 'temperature', label: 'Temperatura', color: palette.observed })
    if (hasModel) {
      legend.push({ id: 'model', label: 'Modelo', color: palette.model, dash: SERIES_DASH.model })
    }
    if (hasPrecipitation) {
      legend.push({ id: 'precipitation', label: 'Precipitación', color: palette.rain })
    }
    if (hasInterval) {
      const level =
        intervalLevel !== null ? ` ${formatPercent(intervalLevel * 100, 0)}` : ''
      legend.push({ id: 'band', label: `Banda${level}`, color: palette.model })
    }
  } else if (variable === 'precipitation') {
    legend.push({ id: 'precipitation', label: 'Precipitación', color: palette.rain })
  } else {
    legend.push({ id: 'wind', label: 'Viento', color: palette.observed })
  }
  const legendItems = legend.map((item) => ({
    ...item,
    active: isVisible((item.id ?? 'temperature') as SeriesKey),
  }))

  /* `view`/`bounds` are only null while there is no data at all. */
  if (!bounds || !view) {
    return (
      <div style={{ minHeight: height }} className="flex items-center justify-center">
        <EmptyState
          title="Sin observaciones en el rango"
          description="Amplía el rango temporal o comprueba que el backfill ha cargado datos históricos."
        />
      </div>
    )
  }

  const axisProps = chartAxisProps(palette)

  const intervalRow = (entries: TooltipEntry[]) => {
    if (!hasInterval || !isVisible('band')) return null
    const row = entries[0]?.payload as HistoryChartRow | undefined
    if (!row || row.band_base === null || row.band_size === null) return null
    const upper = row.band_base + row.band_size
    return (
      <div className="flex items-center justify-between gap-3">
        <span className="text-label text-fg-2">Banda</span>
        <span className="nums text-label font-medium text-fg">
          {formatMetric(row.band_base, 'temperature', units)} –{' '}
          {formatMetric(upper, 'temperature', units)}
        </span>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-1">
        <div className="flex items-center gap-1" role="group" aria-label="Zoom del gráfico">
          <Button
            variant="ghost"
            size="icon"
            className="h-11 w-11"
            aria-label="Alejar"
            onClick={zoom.zoomOut}
            disabled={!zoom.zoomed}
          >
            <Minus className="h-4 w-4" aria-hidden />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-11 w-11"
            aria-label="Acercar"
            onClick={zoom.zoomIn}
            disabled={!zoom.canZoomIn}
          >
            <Plus className="h-4 w-4" aria-hidden />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-11 px-3"
            onClick={zoom.reset}
            disabled={!zoom.zoomed}
          >
            Restablecer
          </Button>
        </div>
        <p className="text-label text-fg-3">
          Rueda para zoom · arrastra para mover · doble clic para restablecer
        </p>
      </div>

      {/* Pointer surface: touch-action pan-y keeps the page scrolling
          vertically on touch, horizontal drags pan the chart. */}
      <div
        ref={zoom.surfaceRef}
        data-testid="chart-surface"
        className="cursor-grab touch-pan-y select-none active:cursor-grabbing"
        style={{ height }}
        {...zoom.handlers}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={CHART_MARGIN}>
            <CartesianGrid {...chartGridProps(palette)} />
            <XAxis
              dataKey="ts"
              type="number"
              scale="time"
              domain={view}
              allowDataOverflow
              minTickGap={48}
              tickFormatter={(value: number) => formatDateTimeMs(value)}
              {...axisProps}
            />

            {variable === 'temperature' ? (
              <>
                <YAxis
                  yAxisId="value"
                  domain={temperatureDomain}
                  width={AXIS_LEFT}
                  tickFormatter={(value: number) => formatMetricTick(value, 'temperature')}
                  label={chartAxisLabel(palette, unitLabel, 'insideLeft')}
                  {...axisProps}
                />
                <YAxis
                  yAxisId="rain"
                  orientation="right"
                  domain={[0, precipitationMax]}
                  width={AXIS_RIGHT_RAIN}
                  tickFormatter={(value: number) => formatMetricTick(value, 'precipitation')}
                  label={chartAxisLabel(palette, metricUnit('precipitation'), 'insideRight')}
                  {...axisProps}
                />
              </>
            ) : (
              <YAxis
                yAxisId="value"
                domain={variable === 'precipitation' ? [0, precipitationMax] : windDomain}
                width={AXIS_LEFT}
                tickFormatter={(value: number) => formatMetricTick(value, variable)}
                label={chartAxisLabel(palette, metricUnit(variable, units), 'insideLeft')}
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
                    const metric = metricForSeries(entry.dataKey)
                    /* Precipitation rows stay in mm: history is not converted
                       on build, so the label must not claim inches. */
                    return formatMetric(
                      entry.value as number,
                      metric,
                      metric === 'precipitation' ? 'metric' : units,
                    )
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
                type="linear"
                dataKey="model"
                name="Modelo"
                stroke={palette.model}
                strokeWidth={1.5}
                strokeDasharray={SERIES_DASH.model}
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
                fill={palette.rain}
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
                fill={palette.rain}
                fillOpacity={0.6}
                isAnimationActive={false}
                maxBarSize={14}
              />
            ) : null}

            {variable === 'temperature' && isVisible('temperature') ? (
              <Line
                yAxisId="value"
                type="linear"
                dataKey="temperature"
                name="Temperatura"
                stroke={palette.observed}
                strokeWidth={1.5}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            ) : null}

            {variable === 'wind' && isVisible('wind') ? (
              <Line
                yAxisId="value"
                type="linear"
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
