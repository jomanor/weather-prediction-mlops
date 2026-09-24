import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

type ReadoutSize = 'sm' | 'md' | 'lg' | 'hero'

const VALUE_SIZES: Record<ReadoutSize, string> = {
  sm: 'text-sm',
  md: 'text-base',
  lg: 'text-2xl',
  hero: 'text-4xl',
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
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className={cn('nums font-medium leading-none text-fg', VALUE_SIZES[size])}>
          {value}
        </span>
        {unit ? <span className="nums text-xs text-fg-3">{unit}</span> : null}
        {trailing ? <span className="ml-auto self-center">{trailing}</span> : null}
      </div>
      {hint ? <div className="mt-1 text-[11px] text-fg-3">{hint}</div> : null}
    </div>
  )
}
