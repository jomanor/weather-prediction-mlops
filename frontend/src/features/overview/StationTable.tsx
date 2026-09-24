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

interface StationTableProps {
  stations: CurrentWeather[]
  predictions: Map<string, Prediction>
  selectedCity?: string | null
}

export function StationTable({ stations, predictions, selectedCity }: StationTableProps) {
  const navigate = useNavigate()
  const { units } = usePreferences()

  if (!stations.length) {
    return (
      <EmptyState
        title="Sin estaciones"
        description="La API no ha devuelto observaciones. Comprueba que el consumidor de Kafka está escribiendo en MongoDB."
      />
    )
  }

  const sorted = [...stations].sort((a, b) =>
    a.city.localeCompare(b.city, 'es', { sensitivity: 'base' }),
  )

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
          {sorted.map((station) => {
            const weather = describeWeather(station.weather_code)
            const Icon = weather.icon
            const prediction = predictions.get(station.city)
            const predicted = prediction?.predicted_temperature ?? null
            const delta =
              isNum(predicted) && isNum(station.temperature) ? predicted - station.temperature : null

            return (
              <tr
                key={station.city}
                onClick={() => navigate(`/stations?city=${encodeURIComponent(station.city)}`)}
                className={cn(
                  'group cursor-pointer border-b border-line/60 transition-colors last:border-0 hover:bg-panel-2',
                  station.city === selectedCity && 'bg-panel-2',
                )}
              >
                <td className="whitespace-nowrap px-4 py-2">
                  <div className="flex items-center gap-2.5">
                    <Icon className={cn('h-4 w-4 shrink-0', toneClass[weather.tone])} />
                    <span>
                      <span className="block font-medium text-fg">{station.city}</span>
                      <span className="block text-[10px] text-fg-3">{weather.label}</span>
                    </span>
                    <ArrowUpRight className="h-3 w-3 shrink-0 text-fg-3 opacity-0 transition-opacity group-hover:opacity-100" />
                  </div>
                </td>
                <td className="nums whitespace-nowrap px-4 py-2 text-right text-fg">
                  {formatTemperature(station.temperature, units)}
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
                  {formatPercent(station.humidity)}
                </td>
                <td className="nums whitespace-nowrap px-4 py-2 text-right text-fg-2">
                  {formatWind(station.wind_speed, units)}
                  <span className="ml-1.5 text-fg-3">{compassPoint(station.wind_direction)}</span>
                </td>
                <td className="nums whitespace-nowrap px-4 py-2 text-right text-fg-2">
                  {formatPressure(station.pressure, units)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
