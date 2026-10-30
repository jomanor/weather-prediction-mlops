import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

/**
 * Four steps on the type scale, nothing larger. `md` is the default readout
 * (14 px); `hero` is the 22 px ceiling for a single emphasised value.
 */
type ReadoutSize = 'sm' | 'md' | 'lg' | 'hero'

const VALUE_SIZES: Record<ReadoutSize, string> = {
  sm: 'text-body',
  md: 'text-value',
  lg: 'text-page',
  hero: 'text-hero',
}

interface ReadoutProps {
  label: ReactNode
  value: ReactNode
  unit?: ReactNode
  hint?: ReactNode
  trailing?: ReactNode
  size?: ReadoutSize
  className?: string
}

/** A single readout: quiet label, tabular value, optional unit. */
export function Readout({
  label,
  value,
  unit,
  hint,
  trailing,
  size = 'md',
  className,
}: ReadoutProps) {
  return (
    <div className={cn('min-w-0', className)}>
      <div className="label">{label}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className={cn('nums font-medium leading-none text-fg', VALUE_SIZES[size])}>
          {value}
        </span>
        {unit ? <span className="nums text-body text-fg-3">{unit}</span> : null}
        {trailing ? <span className="ml-auto self-center">{trailing}</span> : null}
      </div>
      {hint ? <div className="mt-1 text-label text-fg-3">{hint}</div> : null}
    </div>
  )
}
