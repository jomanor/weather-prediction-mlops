import { CloudOff, RefreshCw } from 'lucide-react'
import { lazy, Suspense, useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { useCities } from '@/api/queries'
import { StatusDot } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { ErrorState, LoadingBlock, MapSkeleton, TableSkeleton } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { useMapStations, useStations } from '@/features/overview/queries'
import { StationTable } from '@/features/overview/StationTable'
import { FRESH_WINDOW_SECONDS, useFreshness, type ServerAge } from '@/hooks/useFreshness'
import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'
import { useStationSelection } from '@/hooks/useStationSelection'
import { formatTime } from '@/lib/format'

const StationMap = lazy(() =>
  import('@/components/map/StationMap').then((module) => ({ default: module.StationMap })),
)

/**
 * Resumen: the whole network at a glance — map, one freshness line and the
 * compact station list. Reading it costs three endpoints (`/cities`,
 * `/weather/current`, `/map/stations`); quality, models, benchmark and the
 * per-city predictions live in Sistema and Estación.
 *
 * A pin or a row opens Estación; there is no sheet and no in-page selection.
 */
export function OverviewPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { selectedCity } = useStationSelection()

  const citiesQuery = useCities()
  const stationsQuery = useStations()
  const mapStationsQuery = useMapStations()

  const cities = citiesQuery.data ?? []
  const stations = stationsQuery.data?.stations ?? []
  const stationIndex = useMemo(
    () => new Map(stations.map((station) => [station.city, station])),
    [stations],
  )

  /* Open Estación for one station, keeping the params this surface carries. */
  const openStation = (name: string) => {
    const params = new URLSearchParams(location.search)
    params.set('city', name)
    navigate({ pathname: '/stations', search: params.toString() })
  }

  /* The freshest observation on the network is what the one-liner reports; the
     age window matches the real ingest cadence (see `useFreshness`). */
  const newest = useMemo(
    () =>
      stations.reduce<string | null>(
        (latest, station) =>
          !latest || new Date(station.observed_at) > new Date(latest) ? station.observed_at : latest,
        null,
      ),
    [stations],
  )
  /* The API's `X-Data-Age-Seconds` can only raise the freshness estimate, never
     lower it: `/weather/current` is cacheable, so a stalled pipeline keeps
     answering 304 with the stored header replayed and `ageCapturedAt` reset,
     which would otherwise freeze this value while the data ages. `useFreshness`
     keeps the larger of it and the `observed_at` math below. */
  const dataAge = stationsQuery.dataAgeSeconds
  const ageCapturedAt = stationsQuery.ageCapturedAt
  const serverAge = useMemo<ServerAge | null>(
    () =>
      dataAge === null || ageCapturedAt === null
        ? null
        : { ageSeconds: dataAge, receivedAt: ageCapturedAt },
    [dataAge, ageCapturedAt],
  )
  const { ageSeconds, isFresh } = useFreshness(newest, FRESH_WINDOW_SECONDS, serverAge)
  /* Keyed on the derived age, not on `newest`: the age can exist (from the
     server) while there is no payload timestamp to compare it against, and the
     '>12 h' line must still appear in that case. */
  const stale = ageSeconds !== null && !isFresh
  const reducedMotion = usePrefersReducedMotion()

  if (stationsQuery.isError || citiesQuery.isError) {
    return (
      <div className="p-4 sm:p-6">
        <ErrorState
          title="No se pudo cargar la red de estaciones"
          description={(stationsQuery.error ?? citiesQuery.error)?.message}
          action={
            <Button onClick={() => stationsQuery.refetch()}>
              <RefreshCw className="h-3.5 w-3.5" />
              Reintentar
            </Button>
          }
        />
      </div>
    )
  }

  return (
    <div className="space-y-4 pb-8">
      <PageHeader
        title="Resumen"
        description="Observación en vivo de las estaciones monitorizadas."
        actions={
          <Button onClick={() => stationsQuery.refetch()} disabled={stationsQuery.isFetching}>
            <RefreshCw className={stationsQuery.isFetching ? 'h-3.5 w-3.5 animate-spin' : 'h-3.5 w-3.5'} />
            Actualizar
          </Button>
        }
      />

      <div className="px-4 sm:px-6">
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-fg-3">
          <StatusDot tone={stale ? 'warn' : isFresh ? 'ok' : 'neutral'} pulse={isFresh && !reducedMotion} />
          <span className="nums">
            {stations.length} estaciones · actualizado a las{' '}
            {newest ? formatTime(newest) : '—'}
          </span>
          {stale ? <span className="text-warn">· datos con más de 12 h</span> : null}
        </p>
      </div>

      <div className="grid items-start gap-4 px-4 sm:px-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Panel flush className="overflow-hidden">
          <PanelHeader
            title="Mapa de estaciones"
            subtitle="Icono por tiempo, color y tamaño por temperatura y viento"
          />
          <div className="h-[320px] sm:h-[420px] xl:h-[560px]">
            {mapStationsQuery.isLoading ? (
              <MapSkeleton />
            ) : mapStationsQuery.isError ? (
              <ErrorState
                title="No se pudo cargar el mapa"
                description={mapStationsQuery.error.message}
                action={<Button onClick={() => mapStationsQuery.refetch()}>Reintentar</Button>}
              />
            ) : (
              <Suspense fallback={<LoadingBlock label="Cargando mapa…" />}>
                <StationMap
                  collection={mapStationsQuery.data ?? null}
                  selectedCity={selectedCity}
                  onSelect={openStation}
                />
              </Suspense>
            )}
          </div>
        </Panel>

        <Panel flush className="overflow-hidden">
          <PanelHeader
            title="Estaciones"
            subtitle="Abre una estación para ver su evolución y su predicción"
          />
          {stationsQuery.isLoading ? (
            <TableSkeleton />
          ) : stations.length === 0 ? (
            <div className="flex flex-col items-center py-12 text-fg-3">
              <CloudOff className="h-5 w-5" />
              <p className="mt-3 text-xs">Sin observaciones disponibles</p>
            </div>
          ) : (
            <StationTable
              rows={cities.map((city) => ({
                name: city.name,
                station: stationIndex.get(city.name),
              }))}
            />
          )}
        </Panel>
      </div>
    </div>
  )
}
