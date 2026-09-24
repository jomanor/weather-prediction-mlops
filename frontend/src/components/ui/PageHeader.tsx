import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

interface PageHeaderProps {
  eyebrow?: ReactNode
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
  className?: string
}

/** Editorial masthead that opens every view. */
export function PageHeader({ eyebrow, title, description, actions, className }: PageHeaderProps) {
  return (
    <header className={cn('flex flex-wrap items-end justify-between gap-4 px-4 pt-6 sm:px-6', className)}>
      <div className="min-w-0">
        {eyebrow ? <div className="label mb-2 text-accent">{eyebrow}</div> : null}
        <h1 className="text-2xl font-semibold leading-none tracking-[-0.02em] text-fg sm:text-[28px]">
          {title}
        </h1>
        {description ? (
          <p className="mt-2 max-w-2xl text-[13px] leading-relaxed text-fg-2">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  )
}
