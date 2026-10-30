import { cn } from '@/lib/cn'

export interface LegendItem {
  /** Stable identity; defaults to `label`. */
  id?: string
  label: string
  color: string
  /** Dash pattern preview, matching the series line style. */
  dash?: string
  /** Only used when the legend is interactive. */
  active?: boolean
}

interface SeriesLegendProps {
  items: LegendItem[]
  className?: string
  /** When provided the legend becomes a row of toggles (`aria-pressed`). */
  onToggle?: (id: string) => void
  /** Accessible name for the toggle group. */
  label?: string
}

/**
 * Chart series legend. Static by default; with `onToggle` every entry is a
 * real button, so hiding a series is reachable by keyboard and its state is
 * announced (`aria-pressed`) instead of being colour-only.
 */
export function SeriesLegend({ items, className, onToggle, label }: SeriesLegendProps) {
  const interactive = Boolean(onToggle)

  return (
    <ul
      className={cn('flex flex-wrap items-center gap-x-3 gap-y-1', className)}
      aria-label={interactive ? label : undefined}
    >
      {items.map((item) => {
        const id = item.id ?? item.label
        const active = item.active ?? true
        const content = (
          <>
            <svg
              width="16"
              height="8"
              aria-hidden
              className={cn('shrink-0', !active && 'opacity-40')}
            >
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
            <span className={cn('text-label', active ? 'text-fg-2' : 'text-fg-3 line-through')}>
              {item.label}
            </span>
          </>
        )

        return (
          <li key={id} className="flex items-center">
            {onToggle ? (
              <button
                type="button"
                aria-pressed={active}
                onClick={() => onToggle(id)}
                className={cn(
                  'motion-ui flex cursor-pointer items-center gap-2 rounded-chip border px-1.5 py-1 text-left',
                  'focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
                  /* Pressed series stay readable; a hidden one greys out but the
                     button keeps its hit area so it can be restored. */
                  active
                    ? 'border-line bg-panel-2 hover:border-line-strong'
                    : 'border-transparent hover:bg-panel-2',
                )}
              >
                {content}
              </button>
            ) : (
              <span className="flex items-center gap-2 px-1 py-1">{content}</span>
            )}
          </li>
        )
      })}
    </ul>
  )
}
