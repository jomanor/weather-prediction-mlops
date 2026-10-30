import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react'

import type { SegmentedOption } from '@/components/ui/Segmented'
import { cn } from '@/lib/cn'

/** Same option shape as `Segmented`, so a view switcher can become tabs. */
export type TabOption<T extends string> = SegmentedOption<T>

interface TabsProps<T extends string> {
  value: T
  onChange: (value: T) => void
  options: readonly TabOption<T>[]
  /** Accessible name for the tablist, e.g. "Secciones de Sistema". */
  label: string
  /** The panel for the selected tab; it receives `role="tabpanel"`. */
  children: ReactNode
  className?: string
}

/**
 * Horizontal tabs on the ARIA tabs pattern: one tab stop for the whole list,
 * arrows move *and* select, Home/End jump to the ends, `aria-selected` carries
 * the state and the active tab is linked to its panel both ways.
 *
 * The tab strip is the only navigation control here; it never duplicates the
 * app's own nav (Sistema owns one URL param, not a nested stack).
 */
export function Tabs<T extends string>({
  value,
  onChange,
  options,
  label,
  children,
  className,
}: TabsProps<T>) {
  const baseId = useId()
  const buttons = useRef<(HTMLButtonElement | null)[]>([])
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  )

  const tabId = (option: TabOption<T>) => `${baseId}-tab-${option.value}`
  const panelId = `${baseId}-panel`

  const move = (index: number) => {
    const next = (index + options.length) % options.length
    onChange(options[next].value)
    buttons.current[next]?.focus()
  }

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === 'ArrowRight') {
      event.preventDefault()
      move(index + 1)
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault()
      move(index - 1)
    } else if (event.key === 'Home') {
      event.preventDefault()
      move(0)
    } else if (event.key === 'End') {
      event.preventDefault()
      move(options.length - 1)
    }
  }

  return (
    <div className={cn('min-w-0', className)}>
      <div
        role="tablist"
        aria-label={label}
        aria-orientation="horizontal"
        className="flex items-stretch gap-0 border-b border-line"
      >
        {options.map((option, index) => {
          const active = option.value === value
          return (
            <button
              key={String(option.value)}
              ref={(node) => {
                buttons.current[index] = node
              }}
              id={tabId(option)}
              type="button"
              role="tab"
              aria-selected={active}
              aria-controls={panelId}
              tabIndex={index === selectedIndex ? 0 : -1}
              title={option.title}
              onClick={() => onChange(option.value)}
              onKeyDown={(event) => onKeyDown(event, index)}
              className={cn(
                '-mb-px border-b-2 px-3 py-2 text-title font-medium',
                'motion-ui focus-visible:outline-2 focus-visible:outline-accent',
                active
                  ? 'border-fg text-fg'
                  : 'border-transparent text-fg-3 hover:border-line-strong hover:text-fg-2',
              )}
            >
              {option.label}
            </button>
          )
        })}
      </div>
      <div
        id={panelId}
        role="tabpanel"
        aria-labelledby={tabId(options[selectedIndex])}
        tabIndex={0}
        className="focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        {children}
      </div>
    </div>
  )
}
