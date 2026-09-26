import type { CurrentWeather } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { StatusDot } from '@/components/ui/Badge'
import { Readout } from '@/components/ui/Readout'
import { Panel } from '@/components/ui/Panel'
import { useFreshness } from '@/hooks/useFreshness'
import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'
import { cn } from '@/lib/cn'
import {
  compassPoint,
  formatDateTime,
  formatPercent,
  formatPrecipitation,
  formatPressure,
  formatRelative,
  formatTemperature,
  formatWind,
  isNum,
} from '@/lib/format'
import { describeWeather, toneClass } from '@/lib/weather'

export function CurrentConditions({ station }: { station: CurrentWeather }) {
  const { units } = usePreferences()
  const { isFresh } = useFreshness(station.observed_at)
  const reducedMotion = usePrefersReducedMotion()
  const weather = describeWeather(station.weather_code)
  const WeatherIcon = weather.icon

  return (
    <Panel className="overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="label">Observación</div>
          <h2 className="mt-1.5 text-lg font-semibold tracking-tight text-fg">{station.city}</h2>
          <p className="nums mt-1 text-[11px] text-fg-3">
            {isNum(station.latitude) ? `${station.latitude.toFixed(2)}°N` : '—'} ·{' '}
            {isNum(station.longitude) ? `${station.longitude.toFixed(2)}°E` : '—'}
          </p>
        </div>
        <div className="text-right">
          <div className="label">Registrado</div>
          <div className="nums mt-1.5 text-xs text-fg-2">{formatDateTime(station.observed_at)}</div>
          <div className="mt-0.5 flex items-center justify-end gap-1.5 text-[11px] text-fg-3">
            <StatusDot tone={isFresh ? 'ok' : 'neutral'} pulse={isFresh && !reducedMotion} />
            {formatRelative(station.observed_at)}
          </div>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap items-end gap-x-8 gap-y-4 border-b border-line pb-5">
        <div className="flex items-center gap-4">
          <WeatherIcon className={cn('h-10 w-10', toneClass[weather.tone])} strokeWidth={1.5} />
          <div>
            <div className="nums text-5xl font-medium leading-none tracking-[-0.03em] text-fg">
              {formatTemperature(station.temperature, units).replace('°', '')}
              <span className="text-fg-3">°</span>
            </div>
            <div className="mt-1.5 text-xs text-fg-2">{weather.label}</div>
          </div>
        </div>

        <div className="grid min-w-0 flex-1 grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-3">
          <Readout
            size="sm"
            label="Sensación"
            value={formatTemperature(station.apparent_temperature, units)}
          />
          <Readout size="sm" label="Humedad" value={formatPercent(station.humidity)} />
          <Readout
            size="sm"
            label="Viento"
            value={formatWind(station.wind_speed, units)}
            hint={compassPoint(station.wind_direction)}
          />
          <Readout size="sm" label="Presión" value={formatPressure(station.pressure, units)} />
          <Readout
            size="sm"
            label="Precipitación"
            value={formatPrecipitation(station.precipitation, units)}
          />
          <Readout size="sm" label="Nubosidad" value={formatPercent(station.cloud_cover)} />
        </div>
      </div>
    </Panel>
  )
}
