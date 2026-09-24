import { RefreshCw } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { useCities, useCurrentWeather, useHistory, usePredictions } from '@/api/queries'
import { usePreferences } from '@/app/preferences'
import { HistoryChart } from '@/components/charts/HistoryChart'
import { SeriesLegend } from '@/components/charts/SeriesLegend'
import { Button } from '@/components/ui/Button'
import { EmptyState, ErrorState, LoadingBlock } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Segmented } from '@/components/ui/Segmented'
import { CurrentConditions } from '@/features/station/CurrentConditions'
import { DerivedPanel } from '@/features/station/DerivedPanel'
import { LatestPrediction } from '@/features/station/LatestPrediction'
import { cn } from '@/lib/cn'

const RANGES = [
  { value: 24, label: '24 h' },
  { value: 48, label: '48 h' },
  { value: 72, label: '72 h' },
  { value: 168, label: '7 d' },
]

export function StationPage() {
  const [params, setParams] = useSearchParams()
  const { palette } = usePreferences()
  const { data: cityList } = useCities()
  const cities = useMemo(() => cityList?.map((c) => c.name) ?? [], [cityList])
  const [hours, setHours] = useState(48)

  const city = params.get('city') ?? cities?.[0] ?? ''
  const current = useCurrentWeather(city)
  const history = useHistory(city, hours)
  const predictions = usePredictions(city, 24)

  const selectCity = (next: string) => {
    const updated = new URLSearchParams(params)
    updated.set('city', next)
    setParams(updated, { replace: true })
  }

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
                    <SeriesLegend
                      className="hidden sm:flex"
                      items={[
                        { label: 'Temperatura', color: palette.aemet },
                        { label: 'Precipitación', color: palette.accent },
                      ]}
                    />
                    <Segmented value={hours} onChange={setHours} options={RANGES} label="Rango" />
                  </>
                }
              />
              <div className="p-3">
                {history.isLoading ? (
                  <LoadingBlock />
                ) : history.isError ? (
                  <ErrorState title="Sin histórico" description={history.error.message} />
                ) : !history.data?.points.length ? (
                  <EmptyState
                    title="Sin observaciones en el rango"
                    description="Amplía el rango temporal o comprueba que el backfill ha cargado datos históricos."
                  />
                ) : (
                  <HistoryChart points={history.data.points} palette={palette} />
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
