import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

interface PanelProps {
  children: ReactNode
  className?: string
  /** Removes the inner padding, e.g. for tables or full-bleed charts. */
  flush?: boolean
}

export function Panel({ children, className, flush = false }: PanelProps) {
  return (
    <section
      className={cn(
        'panel-shadow hairline rounded-[var(--radius-panel)] border border-line bg-panel',
        !flush && 'p-4',
        className,
      )}
    >
      {children}
    </section>
  )
}

interface PanelHeaderProps {
  title: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  className?: string
}

export function PanelHeader({ title, subtitle, actions, className }: PanelHeaderProps) {
  return (
    <header
      className={cn(
        'flex flex-wrap items-start justify-between gap-3 border-b border-line px-4 py-3',
        className,
      )}
    >
      <div className="min-w-0">
        <h2 className="text-sm font-semibold tracking-tight text-fg">{title}</h2>
        {subtitle ? <p className="mt-0.5 text-xs text-fg-3">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-1.5">{actions}</div> : null}
    </header>
  )
}

export function PanelBody({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('p-4', className)}>{children}</div>
}
