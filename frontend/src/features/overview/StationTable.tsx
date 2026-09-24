import { ArrowUpRight } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import type { CurrentWeather, Prediction } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { EmptyState } from '@/components/ui/Feedback'
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
  selectedCity?: string | null
}

export function StationTable({ rows, predictions, selectedCity }: StationTableProps) {
  const navigate = useNavigate()
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

            return (
              <tr
                key={row.name}
                onClick={() => navigate(`/stations?city=${encodeURIComponent(row.name)}`)}
                className={cn(
                  'group cursor-pointer border-b border-line/60 transition-colors last:border-0 hover:bg-panel-2',
                  row.name === selectedCity && 'bg-panel-2',
                )}
              >
                <td className="whitespace-nowrap px-4 py-2">
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
                      <span className="block font-medium text-fg">{row.name}</span>
                      <span className="block text-[10px] text-fg-3">
                        {weather?.label ?? 'esperando ingesta'}
                      </span>
                    </span>
                    <ArrowUpRight className="h-3 w-3 shrink-0 text-fg-3 opacity-0 transition-opacity group-hover:opacity-100" />
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
