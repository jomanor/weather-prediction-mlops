import type { ReactNode } from 'react'

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
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn(
        'inline-flex items-center gap-0.5 rounded-md border border-line bg-panel-2 p-0.5',
        className,
      )}
    >
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={String(option.value)}
            type="button"
            role="radio"
            aria-checked={active}
            title={option.title}
            onClick={() => onChange(option.value)}
            className={cn(
              'nums rounded px-2 py-1 text-[11px] font-medium transition-colors',
              active ? 'bg-panel text-fg shadow-sm' : 'text-fg-3 hover:text-fg-2',
            )}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
