import { useRef, type KeyboardEvent, type ReactNode } from 'react'

import { cn } from '@/lib/cn'

export interface SegmentedOption<T extends string | number> {
  value: T
  label: ReactNode
  title?: string
}

interface SegmentedProps<T extends string | number> {
  value: T
  onChange: (value: T) => void
  options: readonly SegmentedOption<T>[]
  label?: string
  className?: string
}

export function Segmented<T extends string | number>({
  value,
  onChange,
  options,
  label,
  className,
}: SegmentedProps<T>) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([])
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  )

  /* ARIA radiogroup pattern: arrows move and select, Home/End jump to the
   * ends, and exactly one option stays in the tab order. */
  const move = (index: number) => {
    const next = (index + options.length) % options.length
    onChange(options[next].value)
    buttons.current[next]?.focus()
  }

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault()
      move(index + 1)
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
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
    <div
      role="radiogroup"
      aria-label={label}
      className={cn(
        'inline-flex items-stretch rounded-[3px] border border-line bg-panel p-0',
        className,
      )}
    >
      {options.map((option, index) => {
        const active = option.value === value
        return (
          <button
            key={String(option.value)}
            ref={(node) => {
              buttons.current[index] = node
            }}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={index === selectedIndex ? 0 : -1}
            title={option.title}
            onClick={() => onChange(option.value)}
            onKeyDown={(event) => onKeyDown(event, index)}
            className={cn(
              'nums rounded px-2 py-1 text-[11px] font-medium transition-colors',
              active ? 'bg-panel-2 font-medium text-fg' : 'text-fg-3 hover:text-fg-2',
            )}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
