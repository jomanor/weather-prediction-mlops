import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

interface PageHeaderProps {
  title: ReactNode
  /** Muted one-liner shown under the title; keep it to a single sentence. */
  description?: ReactNode
  actions?: ReactNode
  className?: string
}

/** Plain document heading: no eyebrow, no display type, no decoration. */
export function PageHeader({ title, description, actions, className }: PageHeaderProps) {
  return (
    <header
      className={cn(
        'flex flex-wrap items-end justify-between gap-x-6 gap-y-3 px-4 py-5 sm:px-6',
        className,
      )}
    >
      <div className="min-w-0">
        <h1 className="text-page font-semibold leading-tight text-fg">{title}</h1>
        {description ? (
          <p className="mt-1 text-title leading-snug text-fg-2">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  )
}
