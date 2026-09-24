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
}

/** Shared, design-system-styled tooltip for every chart. */
export function ChartTooltip({ active, label, payload, format, title }: TooltipContentProps) {
  if (!active || !payload?.length) return null

  return (
    <div className="panel-shadow min-w-[9rem] rounded-md border border-line bg-panel/95 px-2.5 py-2 backdrop-blur">
      <div className="label mb-1.5">{title ? title(label) : label}</div>
      <ul className="space-y-1">
        {payload.map((entry, index) => (
          <li
            key={`${entry.dataKey ?? entry.name ?? index}`}
            className="flex items-center justify-between gap-3"
          >
            <span className="flex items-center gap-1.5 text-[11px] text-fg-2">
              <span
                aria-hidden
                className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
                style={{ background: entry.color }}
              />
              {entry.name}
            </span>
            <span className={cn('nums text-[11px] font-medium text-fg')}>
              {format ? format(entry) : entry.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
