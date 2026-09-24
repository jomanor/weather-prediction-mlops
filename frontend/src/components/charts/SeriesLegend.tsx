import { cn } from '@/lib/cn'

export interface LegendItem {
  label: string
  color: string
  /** Dash pattern preview, matching the series line style. */
  dash?: string
}

export function SeriesLegend({ items, className }: { items: LegendItem[]; className?: string }) {
  return (
    <ul className={cn('flex flex-wrap items-center gap-x-4 gap-y-1.5', className)}>
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          <svg width="16" height="8" aria-hidden className="shrink-0">
            <line
              x1="0"
              y1="4"
              x2="16"
              y2="4"
              stroke={item.color}
              strokeWidth="2"
              strokeDasharray={item.dash}
              strokeLinecap="round"
            />
          </svg>
          <span className="text-[11px] text-fg-2">{item.label}</span>
        </li>
      ))}
    </ul>
  )
}
