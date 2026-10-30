import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'

import type { CurrentWeather } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { StatusDot } from '@/components/ui/Badge'
import { EmptyState } from '@/components/ui/Feedback'
import { ageInSeconds, FRESH_WINDOW_SECONDS } from '@/hooks/useFreshness'
import { cn } from '@/lib/cn'
import { compassPoint, formatRelative, formatTemperature, formatWind, isNum } from '@/lib/format'
import { describeWeather, toneClass } from '@/lib/weather'

export interface StationRow {
  /** Registry name. Rows exist for every monitored station, even before the
   * producer has ingested any observation for them. */
  name: string
  station?: CurrentWeather
}

interface StationTableProps {
  rows: StationRow[]
}

type SortKey = 'name' | 'temperature'
interface SortState {
  key: SortKey
  dir: 'asc' | 'desc'
}

/**
 * Compact station list: four fields that fit a 360 px phone — station (with its
 * weather icon), temperature, wind and data age.
 *
 * Every row opens Estación. The link is the only interactive element in the row
 * and it stretches over the whole row (`after:inset-0`), so a mouse click
 * anywhere and a keyboard Enter on the link do the same thing.
 */
export function StationTable({ rows }: StationTableProps) {
  const { units } = usePreferences()
  const location = useLocation()
  const [sort, setSort] = useState<SortState>({ key: 'name', dir: 'asc' })

  const toggleSort = (key: SortKey) =>
    setSort((current) =>
      current.key === key
        ? { key, dir: current.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: key === 'name' ? 'asc' : 'desc' },
    )

  /* Keep any params this surface carries; only `city` changes. */
  const stationLink = (name: string) => {
    const params = new URLSearchParams(location.search)
    params.set('city', name)
    return { pathname: '/stations', search: params.toString() }
  }

  const withObservations = rows.filter((row) => row.station)

  if (!rows.length) {
    return (
      <EmptyState
        title="Sin estaciones registradas"
        description="El registro está vacío: la ingesta no tiene ninguna estación que monitorizar."
      />
    )
  }

  const sorted = [...rows].sort((a, b) => {
    if (sort.key === 'temperature') {
      const left = a.station?.temperature
      const right = b.station?.temperature
      /* Stations without an observation sink to the bottom in either direction. */
      if (left === null || left === undefined) return right === null || right === undefined ? 0 : 1
      if (right === null || right === undefined) return -1
      return sort.dir === 'asc' ? left - right : right - left
    }
    const byName = a.name.localeCompare(b.name, 'es', { sensitivity: 'base' })
    return sort.dir === 'asc' ? byName : -byName
  })

  const isFresh = (station?: CurrentWeather) => {
    const age = ageInSeconds(station?.observed_at)
    return age !== null && age <= FRESH_WINDOW_SECONDS
  }

  const sortLabel = (key: SortKey, label: string) => {
    const active = sort.key === key
    return (
      <button
        type="button"
        onClick={() => toggleSort(key)}
        className={cn(
          'inline-flex items-center gap-1 rounded-[2px] font-normal hover:text-fg',
          active ? 'text-fg-2' : 'text-fg-3',
        )}
      >
        {label}
        <span aria-hidden className={cn('text-[9px]', !active && 'opacity-0')}>
          {sort.dir === 'asc' ? '↑' : '↓'}
        </span>
      </button>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-line">
            <th
              aria-sort={
                sort.key === 'name' ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'
              }
              className="label px-3 py-2 text-left font-normal"
            >
              {sortLabel('name', 'Estación')}
            </th>
            <th
              aria-sort={
                sort.key === 'temperature'
                  ? sort.dir === 'asc'
                    ? 'ascending'
                    : 'descending'
                  : 'none'
              }
              className="label px-3 py-2 text-right font-normal"
            >
              {sortLabel('temperature', 'Temp')}
            </th>
            <th className="label hidden px-3 py-2 text-right font-normal sm:table-cell">Viento</th>
            <th className="label px-3 py-2 text-right font-normal">Datos</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const station = row.station
            const weather = station ? describeWeather(station.weather_code) : null
            const Icon = weather?.icon

            return (
              <tr
                key={row.name}
                className="relative border-b border-line/60 transition-colors last:border-0 hover:bg-panel-2"
              >
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2.5">
                    {Icon ? (
                      <Icon className={cn('h-4 w-4 shrink-0', toneClass[weather!.tone])} />
                    ) : (
                      <span
                        aria-hidden="true"
                        className="h-4 w-4 shrink-0 rounded-full border border-dashed border-line-strong"
                      />
                    )}
                    <span className="min-w-0">
                      <Link
                        to={stationLink(row.name)}
                        className="block truncate font-medium text-fg after:absolute after:inset-0 after:content-['']"
                      >
                        {row.name}
                      </Link>
                      <span className="block truncate text-[10px] text-fg-3">
                        {weather?.label ?? 'esperando ingesta'}
                      </span>
                    </span>
                  </div>
                </td>
                <td className="nums whitespace-nowrap px-3 py-2 text-right text-fg">
                  {formatTemperature(station?.temperature ?? null, units)}
                </td>
                <td className="nums hidden whitespace-nowrap px-3 py-2 text-right text-fg-2 sm:table-cell">
                  {formatWind(station?.wind_speed ?? null, units)}
                  <span className="ml-1 text-fg-3">
                    {isNum(station?.wind_direction) ? compassPoint(station!.wind_direction as number) : ''}
                  </span>
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-right">
                  {station ? (
                    <span className="inline-flex items-center gap-1.5 text-[11px] text-fg-3">
                      <StatusDot tone={isFresh(station) ? 'ok' : 'neutral'} pulse={false} />
                      {formatRelative(station.observed_at)}
                    </span>
                  ) : (
                    <span className="text-[11px] text-fg-3">—</span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {withObservations.length < rows.length ? (
        <p className="border-t border-line px-3 py-2 text-[11px] text-fg-3">
          Las estaciones sin observación aparecen vacías hasta la próxima ingesta del productor.
        </p>
      ) : null}
    </div>
  )
}
