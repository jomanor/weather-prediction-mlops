import { useMemo } from 'react'

import { buildWindRose } from '@/features/analytics/transforms'
import type { WindRoseSector } from '@/features/analytics/schemas'
import { cn } from '@/lib/cn'
import { formatInteger, formatNumber, isNum } from '@/lib/format'
import { rampColor, rampDomain, rampToken } from '@/lib/ramps'

/**
 * Wind rose: 16 sectors of 22.5°, one petal per direction of origin. Petal
 * length encodes frequency; fill uses the shared wind ramp to encode mean
 * speed. Hand-rolled SVG so no charting library is pulled in for a single
 * polar view.
 */

const SIZE = 224
const CENTER = SIZE / 2
const RADIUS = 82
const LABEL_RADIUS = RADIUS + 14
const SECTOR_SPAN = 360 / 16
const SECTOR_GAP = 3

interface Point {
  x: number
  y: number
}

/** Angle measured clockwise from north (0° = up). */
function polar(angle: number, radius: number): Point {
  const radians = ((angle - 90) * Math.PI) / 180
  return { x: CENTER + radius * Math.cos(radians), y: CENTER + radius * Math.sin(radians) }
}

function petalPath(angle: number, radius: number): string {
  const half = (SECTOR_SPAN - SECTOR_GAP) / 2
  const start = polar(angle - half, radius)
  const end = polar(angle + half, radius)
  return `M ${CENTER} ${CENTER} L ${start.x} ${start.y} A ${radius} ${radius} 0 0 1 ${end.x} ${end.y} Z`
}

export function WindRose({
  sectors,
  className,
}: {
  sectors: readonly WindRoseSector[]
  className?: string
}) {
  const model = useMemo(() => buildWindRose(sectors), [sectors])
  const domain = rampDomain('wind')

  if (model.total === 0 || model.maxCount === 0) {
    return (
      <p className={cn('px-1 py-6 text-center text-xs text-fg-3', className)}>
        Sin observaciones de viento en la ventana seleccionada.
      </p>
    )
  }

  const ringFractions = [0.25, 0.5, 0.75, 1]

  return (
    <div className={cn('flex flex-col items-center gap-3', className)}>
      <svg
        role="img"
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="h-56 w-full max-w-[15rem]"
        aria-label={`Rosa de los vientos con ${formatInteger(model.total)} observaciones; dirección dominante ${dominantLabel(model.petals)}`}
      >
        {ringFractions.map((fraction) => (
          <circle
            key={fraction}
            cx={CENTER}
            cy={CENTER}
            r={RADIUS * fraction}
            fill="none"
            stroke="var(--line)"
            strokeWidth={0.75}
          />
        ))}
        {ringFractions.map((fraction) => (
          <text
            key={`label-${fraction}`}
            x={CENTER + 3}
            y={CENTER - RADIUS * fraction + 8}
            className="nums"
            fontSize={7}
            fill="var(--fg-3)"
          >
            {formatInteger(model.maxCount * fraction)}
          </text>
        ))}

        {[0, 90, 180, 270].map((angle) => {
          const edge = polar(angle, RADIUS)
          return (
            <line
              key={angle}
              x1={CENTER}
              y1={CENTER}
              x2={edge.x}
              y2={edge.y}
              stroke="var(--line)"
              strokeWidth={0.75}
            />
          )
        })}

        {model.petals
          .filter((petal) => petal.count > 0)
          .map((petal) => {
            const radius = Math.max(4, (petal.count / model.maxCount) * RADIUS)
            return (
              <path
                key={petal.sector}
                d={petalPath(petal.sector * SECTOR_SPAN, radius)}
                fill={rampColor('wind', petal.meanSpeed ?? 0, domain)}
                fillOpacity={0.85}
                stroke="var(--panel)"
                strokeWidth={0.5}
              >
                <title>
                  {`${petal.label} · ${formatInteger(petal.count)} obs · ${
                    isNum(petal.meanSpeed) ? `${formatNumber(petal.meanSpeed, 1)} km/h` : 'sin velocidad'
                  }`}
                </title>
              </path>
            )
          })}

        <circle cx={CENTER} cy={CENTER} r={1.5} fill="var(--fg-3)" />

        {(
          [
            ['N', 0],
            ['E', 90],
            ['S', 180],
            ['O', 270],
          ] as const
        ).map(([label, angle]) => {
          const point = polar(angle, LABEL_RADIUS)
          return (
            <text
              key={label}
              x={point.x}
              y={point.y + 3}
              textAnchor="middle"
              fontSize={9}
              fill="var(--fg-2)"
            >
              {label}
            </text>
          )
        })}
      </svg>

      <div className="flex items-center gap-2 text-[10px] text-fg-3">
        <span className="nums">{formatNumber(domain[0], 0)}</span>
        <span className="flex" aria-hidden>
          {Array.from({ length: 4 }, (_, step) => (
            <span key={step} className="h-2.5 w-4" style={{ background: rampToken('wind', step) }} />
          ))}
        </span>
        <span className="nums">{formatNumber(domain[1], 0)} km/h</span>
      </div>

      <ul className="sr-only">
        {model.petals.map((petal) => (
          <li key={petal.sector}>
            {`${petal.label}: ${formatInteger(petal.count)} observaciones, ${
              isNum(petal.meanSpeed) ? `${formatNumber(petal.meanSpeed, 1)} km/h de media` : 'sin velocidad media'
            }`}
          </li>
        ))}
      </ul>
    </div>
  )
}

function dominantLabel(petals: ReturnType<typeof buildWindRose>['petals']): string {
  return petals.reduce((best, petal) => (petal.count > best.count ? petal : best), petals[0]).label
}
