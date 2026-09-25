import { RefreshCw } from 'lucide-react'
import { useMemo } from 'react'

import { useCities } from '@/api/queries'
import { usePreferences } from '@/app/preferences'
import { HistoryChart, type HistoryVariable } from '@/components/charts/HistoryChart'
import { SeriesLegend } from '@/components/charts/SeriesLegend'
import { Button } from '@/components/ui/Button'
import { EmptyState, ErrorState, LoadingBlock, ChartSkeleton } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Segmented } from '@/components/ui/Segmented'
import { CurrentConditions } from '@/features/station/CurrentConditions'
import { DerivedPanel } from '@/features/station/DerivedPanel'
import { LatestPrediction } from '@/features/station/LatestPrediction'
import {
  PREDICTION_HORIZONS,
  useCurrentWeather,
  useHistory,
  usePredictions,
  type PredictionHorizon,
} from '@/features/station/queries'
import { useChartWindow } from '@/hooks/useChartSync'
import { resolveCity, useStationSelection } from '@/hooks/useStationSelection'
import { urlOption, useUrlState, type UrlCodec } from '@/hooks/useUrlState'
import { cn } from '@/lib/cn'
import { formatPercent, isNum } from '@/lib/format'

const HOURS = [24, 48, 72, 168] as const
type StationHours = (typeof HOURS)[number]

const RANGES: ReadonlyArray<{ value: StationHours; label: string }> = [
  { value: 24, label: '24 h' },
  { value: 48, label: '48 h' },
  { value: 72, label: '72 h' },
  { value: 168, label: '7 d' },
]

const VARIABLES: ReadonlyArray<{ value: HistoryVariable; label: string }> = [
  { value: 'temperature', label: 'Temp' },
  { value: 'precipitation', label: 'Lluvia' },
  { value: 'wind', label: 'Viento' },
]

/* Contract 2: horizon selector over the M2 config; default +1 h. */
const HORIZON_OPTIONS: ReadonlyArray<{ value: PredictionHorizon; label: string }> =
  PREDICTION_HORIZONS.map((value) => ({ value, label: `+${value} h` }))

/* URL state: the station is shared app-wide; range, variable and horizon are local views. */
const STATION_SCHEMA: {
  hours: UrlCodec<StationHours>
  var: UrlCodec<HistoryVariable>
  horizon: UrlCodec<PredictionHorizon>
} = {
  hours: urlOption(HOURS, 48),
  var: urlOption(['temperature', 'precipitation', 'wind'] as const, 'temperature'),
  horizon: urlOption(PREDICTION_HORIZONS, 1),
}

export function StationPage() {
  const [state, setState] = useUrlState(STATION_SCHEMA)
  const { selectedCity, selectCity } = useStationSelection()
  const { palette, units } = usePreferences()
  const { data: cityList } = useCities()
  const cities = useMemo(() => cityList?.map((c) => c.name) ?? [], [cityList])

  const city = resolveCity(selectedCity, cities, cityList !== undefined)
  const hours = state.hours
  const variable = state.var
  const horizon = state.horizon

  const current = useCurrentWeather(city)
  const history = useHistory(city, hours)
  const predictions = usePredictions(city, 24, horizon)
  const intervalLevel =
    predictions.data?.find((prediction) => isNum(prediction.interval_level))?.interval_level ?? null

  /* Share the observed window with the benchmark charts on the same time axis. */
  const { brushed, resetWindow } = useChartWindow(hours, history.data?.points.at(-1)?.observed_at)

  const legendItems =
    variable === 'temperature'
      ? [
          { label: 'Temperatura', color: palette.aemet },
          { label: 'Precipitación', color: palette.accent },
          ...(intervalLevel !== null
            ? [{ label: `Modelo (nivel ${formatPercent(intervalLevel * 100, 0)})`, color: palette.model }]
            : []),
        ]
      : variable === 'precipitation'
        ? [{ label: 'Precipitación', color: palette.accent }]
        : [{ label: `Viento (${units === 'imperial' ? 'mph' : 'km/h'})`, color: palette.observed }]

  return (
    <div className="space-y-5 pb-8">
      <PageHeader
        title={city || 'Selecciona una estación'}
        description="Condiciones observadas, evolución reciente y última predicción del modelo para la estación seleccionada."
        actions={
          <>
            <label className="sr-only" htmlFor="station-select">
              Estación
            </label>
            <select
              id="station-select"
              value={city}
              onChange={(event) => selectCity(event.target.value)}
              className="h-9 rounded-[3px] border border-line bg-panel px-2.5 text-xs text-fg focus:border-accent focus:outline-none"
            >
              {(cities ?? []).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            <Segmented
              value={hours}
              onChange={(value) => setState({ hours: value })}
              options={RANGES}
              label="Rango"
            />
            <Segmented
              value={horizon}
              onChange={(value) => setState({ horizon: value })}
              options={HORIZON_OPTIONS}
              label="Horizonte de predicción"
            />
            <Button onClick={() => current.refetch()} disabled={current.isFetching}>
              <RefreshCw className={cn('h-3.5 w-3.5', current.isFetching && 'animate-spin')} />
              Actualizar
            </Button>
          </>
        }
      />

      {!city ? (
        <div className="px-4 sm:px-6">
          <Panel>
            <LoadingBlock label="Cargando estaciones…" />
          </Panel>
        </div>
      ) : (
        <>
          <div className="px-4 sm:px-6">
            {current.isError ? (
              <Panel>
                <ErrorState
                  title="No se pudo cargar la observación"
                  description={current.error.message}
                  action={<Button onClick={() => current.refetch()}>Reintentar</Button>}
                />
              </Panel>
            ) : current.isLoading || !current.data ? (
              <Panel>
                <LoadingBlock />
              </Panel>
            ) : (
              <CurrentConditions station={current.data} />
            )}
          </div>

          <div className="grid items-start gap-5 px-4 sm:px-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <Panel flush>
              <PanelHeader
                title="Evolución observada"
                subtitle={`Últimas ${hours} horas`}
                actions={
                  <>
                    <SeriesLegend className="hidden sm:flex" items={legendItems} />
                    {brushed ? (
                      <Button size="sm" onClick={resetWindow}>
                        Restablecer zoom
                      </Button>
                    ) : null}
                    <Segmented
                      value={variable}
                      onChange={(value) => setState({ var: value }, { replace: false })}
                      options={VARIABLES}
                      label="Variable"
                    />
                  </>
                }
              />
              <div className="p-3">
                {history.isLoading ? (
                  <ChartSkeleton />
                ) : history.isError ? (
                  <ErrorState title="Sin histórico" description={history.error.message} />
                ) : !history.data?.points.length ? (
                  <EmptyState
                    title="Sin observaciones en el rango"
                    description="Amplía el rango temporal o comprueba que el backfill ha cargado datos históricos."
                  />
                ) : (
                  <HistoryChart
                    points={history.data.points}
                    predictions={predictions.data ?? []}
                    palette={palette}
                    variable={variable}
                  />
                )}
              </div>
            </Panel>

            {predictions.isLoading ? (
              <Panel>
                <LoadingBlock />
              </Panel>
            ) : (
              <LatestPrediction predictions={predictions.data ?? []} />
            )}
          </div>

          <div className="px-4 sm:px-6">
            {history.data?.points.length ? <DerivedPanel points={history.data.points} /> : null}
          </div>
        </>
      )}
    </div>
  )
}
