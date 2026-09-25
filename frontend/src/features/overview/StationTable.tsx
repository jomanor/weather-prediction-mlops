import { ArrowUpRight, Check } from 'lucide-react'
import { Link } from 'react-router-dom'

import type { CurrentWeather, Prediction } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { EmptyState } from '@/components/ui/Feedback'
import { useStationSelection } from '@/hooks/useStationSelection'
import { cn } from '@/lib/cn'
import {
  compassPoint,
  formatPercent,
  formatPressure,
  formatTemperature,
  formatWind,
  isNum,
} from '@/lib/format'
import { describeWeather, toneClass } from '@/lib/weather'

export interface StationRow {
  /** Registry name. Rows exist for every monitored station, even before the
   * producer has ingested any observation for them. */
  name: string
  station?: CurrentWeather
}

interface StationTableProps {
  rows: StationRow[]
  predictions: Map<string, Prediction>
}

/**
 * Selectable station table. Selection is the app-wide URL-backed `selectedCity`,
 * so clicking a row drives the map and the charts; the row is keyboard reachable
 * and selection is signalled by an icon plus `aria-selected`, never colour alone.
 */
export function StationTable({ rows, predictions }: StationTableProps) {
  const { selectedCity, selectCity } = useStationSelection()
  const { units } = usePreferences()

  const withObservations = rows.filter((row) => row.station)

  if (!rows.length) {
    return (
      <EmptyState
        title="Sin estaciones registradas"
        description="Añade estaciones desde «Gestionar estaciones» para poblar el registro."
      />
    )
  }

  const sorted = [...rows].sort((a, b) => a.name.localeCompare(b.name, 'es', { sensitivity: 'base' }))

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-line">
            <th className="label px-4 py-2 text-left font-normal">Estación</th>
            <th className="label px-4 py-2 text-right font-normal">Temp</th>
            <th className="label px-4 py-2 text-right font-normal">Modelo +1h</th>
            <th className="label px-4 py-2 text-right font-normal">Humedad</th>
            <th className="label px-4 py-2 text-right font-normal">Viento</th>
            <th className="label px-4 py-2 text-right font-normal">Presión</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const station = row.station
            const weather = station ? describeWeather(station.weather_code) : null
            const Icon = weather?.icon
            const prediction = predictions.get(row.name)
            const predicted = prediction?.predicted_temperature ?? null
            const delta =
              isNum(predicted) && isNum(station?.temperature)
                ? (predicted as number) - (station?.temperature as number)
                : null
            const selected = row.name === selectedCity

            return (
              <tr
                key={row.name}
                tabIndex={0}
                aria-selected={selected}
                data-selected={selected}
                onClick={() => selectCity(row.name)}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return
                  event.preventDefault()
                  selectCity(row.name)
                }}
                className={cn(
                  'group relative cursor-pointer border-b border-line/60 transition-colors last:border-0 hover:bg-panel-2',
                  selected && 'bg-panel-2',
                )}
              >
                <td className="whitespace-nowrap px-4 py-2">
                  {selected ? (
                    <span aria-hidden className="absolute inset-y-0 left-0 w-0.5 bg-accent" />
                  ) : null}
                  <div className="flex items-center gap-2.5">
                    {Icon ? (
                      <Icon className={cn('h-4 w-4 shrink-0', toneClass[weather!.tone])} />
                    ) : (
                      <span
                        aria-hidden="true"
                        className="h-4 w-4 shrink-0 rounded-full border border-dashed border-line-strong"
                      />
                    )}
                    <span>
                      <span className="flex items-center gap-1.5 font-medium text-fg">
                        {selected ? <Check aria-hidden className="h-3 w-3 text-accent" /> : null}
                        {row.name}
                      </span>
                      <span className="block text-[10px] text-fg-3">
                        {weather?.label ?? 'esperando ingesta'}
                      </span>
                    </span>
                    <Link
                      to={`/stations?city=${encodeURIComponent(row.name)}`}
                      onClick={(event) => event.stopPropagation()}
                      aria-label={`Abrir la estación ${row.name}`}
                      className="ml-0.5 rounded p-0.5 text-fg-3 opacity-0 transition-opacity hover:text-fg group-focus-within:opacity-100 group-hover:opacity-100 focus-visible:opacity-100"
                    >
                      <ArrowUpRight className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                </td>
                <td className="nums whitespace-nowrap px-4 py-2 text-right text-fg">
                  {formatTemperature(station?.temperature ?? null, units)}
                </td>
                <td className="nums whitespace-nowrap px-4 py-2 text-right">
                  {isNum(predicted) ? (
                    <span className="inline-flex items-baseline gap-1.5 text-model">
                      {formatTemperature(predicted, units)}
                      {delta !== null ? (
                        <span className="text-[10px] text-fg-3">
                          {delta >= 0 ? '+' : '−'}
                          {formatTemperature(Math.abs(delta), units, 1).replace('°', '')}
                        </span>
                      ) : null}
                    </span>
                  ) : (
                    <span className="text-fg-3">—</span>
                  )}
                </td>
                <td className="nums whitespace-nowrap px-4 py-2 text-right text-fg-2">
                  {formatPercent(station?.humidity ?? null)}
                </td>
                <td className="nums whitespace-nowrap px-4 py-2 text-right text-fg-2">
                  {formatWind(station?.wind_speed ?? null, units)}
                  <span className="ml-1.5 text-fg-3">
                    {isNum(station?.wind_direction) ? compassPoint(station!.wind_direction as number) : ''}
                  </span>
                </td>
                <td className="nums whitespace-nowrap px-4 py-2 text-right text-fg-2">
                  {formatPressure(station?.pressure ?? null, units)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {withObservations.length < rows.length ? (
        <p className="border-t border-line px-4 py-2 text-[11px] text-fg-3">
          Las estaciones sin observación aparecen vacías hasta la próxima ingesta del productor.
        </p>
      ) : null}
    </div>
  )
}
