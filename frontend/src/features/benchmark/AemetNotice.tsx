import { AlertTriangle, Info } from 'lucide-react'

import type { City } from '@/api/schemas'
import { cn } from '@/lib/cn'
import { isNum } from '@/lib/format'

/**
 * AEMET OpenData only covers Spain. A station on the American continent is not
 * an outage, so the copy must say so and never imply data is merely missing.
 */
export function isOutsideAemetCoverage(city?: Pick<City, 'longitude'>): boolean {
  return Boolean(city && isNum(city.longitude) && city.longitude < -20)
}

interface AemetNoticeProps {
  available: boolean
  error?: string | null
  city?: City
  /** `banner` sits above the charts; `inline` closes the metrics panel. */
  variant?: 'banner' | 'inline'
}

export function AemetNotice({
  available,
  error,
  city,
  variant = 'banner',
}: AemetNoticeProps) {
  if (available) return null

  const outsideCoverage = isOutsideAemetCoverage(city)
  const title = outsideCoverage
    ? 'AEMET no cubre estaciones de EE. UU.'
    : 'Benchmark AEMET no disponible'
  const description = outsideCoverage
    ? 'Esta estación está fuera de la red de AEMET OpenData (España). Solo se compara la observación con el modelo Spark GBT; no se inventan datos de AEMET.'
    : (error ?? 'La API de AEMET OpenData no ha devuelto datos para esta estación.')

  if (variant === 'inline') {
    return (
      <div className="flex items-start gap-2 border-t border-line px-4 py-3 text-[11px] leading-relaxed">
        <Info className="mt-0.5 h-3 w-3 shrink-0 text-warn" />
        <div>
          <p className="font-medium text-fg">{title}</p>
          <p className="mt-0.5 text-fg-3">{description}</p>
        </div>
      </div>
    )
  }

  return (
    <div
      className={cn(
        'flex items-start gap-2.5 rounded-[var(--radius-panel)] border border-warn/35 bg-warn/5 px-4 py-3',
      )}
      role="status"
    >
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" />
      <div className="text-xs leading-relaxed">
        <p className="font-medium text-fg">{title}</p>
        <p className="mt-0.5 text-fg-2">{description}</p>
      </div>
    </div>
  )
}
