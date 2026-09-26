import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { useEffect, useRef, type PointerEvent as ReactPointerEvent } from 'react'

import type { CurrentWeather, Prediction } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { Button } from '@/components/ui/Button'
import { Readout } from '@/components/ui/Readout'
import {
  compassPoint,
  formatPercent,
  formatPrecipitation,
  formatPressure,
  formatTemperature,
  formatWind,
  isNum,
} from '@/lib/format'
import { describeWeather, toneClass } from '@/lib/weather'

interface StationSheetProps {
  city: string | null
  station?: CurrentWeather
  prediction?: Prediction
  position: { index: number; total: number } | null
  onPrev: () => void
  onNext: () => void
  onClose: () => void
}

const SWIPE_THRESHOLD_PX = 60

/**
 * U6 mobile station detail. A non-modal bottom sheet that sits above the bottom
 * navigation (which keeps working) and swipes horizontally between stations.
 * Escape and the close button dismiss it; the prev/next buttons keep it
 * keyboard reachable.
 */
export function StationSheet({
  city,
  station,
  prediction,
  position,
  onPrev,
  onNext,
  onClose,
}: StationSheetProps) {
  const { units } = usePreferences()
  const startX = useRef<number | null>(null)

  useEffect(() => {
    if (!city) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [city, onClose])

  if (!city) return null

  const weather = station ? describeWeather(station.weather_code) : null
  const WeatherIcon = weather?.icon

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    startX.current = event.clientX
  }
  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (startX.current === null) return
    const delta = event.clientX - startX.current
    startX.current = null
    if (Math.abs(delta) < SWIPE_THRESHOLD_PX) return
    if (delta < 0) onNext()
    else onPrev()
  }

  return (
    <div
      role="dialog"
      aria-label={`Detalle de la estación ${city}`}
      className="fixed inset-x-0 bottom-12 z-30 animate-in border-t border-line bg-panel shadow-[0_-8px_24px_rgb(0_0_0/0.12)] md:hidden"
      style={{ touchAction: 'pan-y' }}
      onPointerDown={onPointerDown}
      onPointerUp={onPointerUp}
    >
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <span aria-hidden className="h-1 w-6 shrink-0 rounded-full bg-line-strong" />
          <h2 className="truncate text-sm font-semibold text-fg">{city}</h2>
          {position ? (
            <span className="nums shrink-0 text-[10px] text-fg-3">
              {position.index}/{position.total}
            </span>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button size="icon" aria-label="Estación anterior" onClick={onPrev}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button size="icon" aria-label="Estación siguiente" onClick={onNext}>
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Button size="icon" aria-label="Cerrar detalle" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-3 px-4 py-3">
        {WeatherIcon ? (
          <WeatherIcon className={toneClass[weather!.tone]} strokeWidth={1.5} />
        ) : (
          <span aria-hidden className="h-6 w-6 rounded-full border border-dashed border-line-strong" />
        )}
        <div className="min-w-0">
          <div className="nums text-2xl font-medium leading-none text-fg">
            {formatTemperature(station?.temperature ?? null, units)}
          </div>
          <div className="mt-1 truncate text-[11px] text-fg-3">
            {weather?.label ?? 'esperando ingesta'}
          </div>
        </div>
        {prediction ? (
          <div className="ml-auto text-right">
            <div className="label">Modelo +{prediction.horizon_hours} h</div>
            <div className="nums mt-1 text-sm text-model">
              {formatTemperature(prediction.predicted_temperature, units)}
            </div>
          </div>
        ) : null}
      </div>

      <div className="grid grid-cols-3 gap-x-4 gap-y-3 border-t border-line px-4 py-3">
        <Readout size="sm" label="Sensación" value={formatTemperature(station?.apparent_temperature ?? null, units)} />
        <Readout size="sm" label="Humedad" value={formatPercent(station?.humidity ?? null)} />
        <Readout
          size="sm"
          label="Viento"
          value={formatWind(station?.wind_speed ?? null, units)}
          hint={isNum(station?.wind_direction) ? compassPoint(station!.wind_direction as number) : undefined}
        />
        <Readout size="sm" label="Presión" value={formatPressure(station?.pressure ?? null, units)} />
        <Readout
          size="sm"
          label="Precipitación"
          value={formatPrecipitation(station?.precipitation ?? null, units)}
        />
        <Readout size="sm" label="Nubosidad" value={formatPercent(station?.cloud_cover ?? null)} />
      </div>
    </div>
  )
}
