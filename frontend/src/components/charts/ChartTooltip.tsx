import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

export interface TooltipEntry {
  name?: string
  value?: number | string | null
  color?: string
  dataKey?: string | number
  payload?: Record<string, unknown>
}

export interface TooltipContentProps {
  active?: boolean
  label?: string | number
  payload?: TooltipEntry[]
  format?: (entry: TooltipEntry) => ReactNode
  title?: (label: string | number | undefined) => ReactNode
  /** Extra rows below the series list, e.g. the interval band. */
  extra?: (payload: TooltipEntry[]) => ReactNode
}

/**
 * The one tooltip shell: hairline border, single elevation, label in the
 * `label` style, values in tabular numerals. Every chart uses it, so hovering
 * any plot looks the same.
 */
export function ChartTooltip({ active, label, payload, format, title, extra }: TooltipContentProps) {
  if (!active || !payload?.length) return null

  /* Only draw the divider when the chart actually produced extra content. */
  const extraContent = extra?.(payload)

  return (
    <div className="shadow-ui min-w-36 rounded-control border border-line bg-panel px-3 py-2">
      <div className="label mb-2">{title ? title(label) : label}</div>
      <ul className="space-y-1">
        {payload.map((entry, index) => (
          <li
            key={`${entry.dataKey ?? entry.name ?? index}`}
            className="flex items-center justify-between gap-3"
          >
            <span className="flex items-center gap-2 text-label text-fg-2">
              <span
                aria-hidden
                className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ background: entry.color }}
              />
              {entry.name}
            </span>
            <span className={cn('nums text-label font-medium text-fg')}>
              {format ? format(entry) : entry.value}
            </span>
          </li>
        ))}
      </ul>
      {extraContent ? (
        <div className="mt-2 border-t border-line pt-2">{extraContent}</div>
      ) : null}
    </div>
  )
}
