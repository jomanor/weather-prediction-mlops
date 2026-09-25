import { AlertTriangle, Inbox, Loader2, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'
import { cn } from '@/lib/cn'

/**
 * U7: shimmer skeleton. The shimmer is a CSS animation gated by the OS motion
 * preference (reduce → a static block, no animation).
 */
export function Skeleton({ className }: { className?: string }) {
  const reduced = usePrefersReducedMotion()
  return (
    <div aria-hidden="true" className={cn('rounded bg-panel-2', !reduced && 'shimmer', className)} />
  )
}

/** Chart placeholder: a title line, a plot block and an axis strip. */
export function ChartSkeleton({ height = 260, className }: { height?: number; className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={cn('flex flex-col gap-2 p-3', className)}
      style={{ height }}
    >
      <Skeleton className="h-3 w-1/3" />
      <Skeleton className="min-h-0 flex-1" />
      <div className="flex gap-2">
        <Skeleton className="h-2 flex-1" />
        <Skeleton className="h-2 flex-1" />
        <Skeleton className="h-2 flex-1" />
      </div>
    </div>
  )
}

/** Table placeholder: evenly spaced rows, sized like the real table. */
export function TableSkeleton({ rows = 6, className }: { rows?: number; className?: string }) {
  return (
    <div aria-hidden="true" className={cn('space-y-2 p-4', className)}>
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} className="h-6 w-full" />
      ))}
    </div>
  )
}

/** Map placeholder: a graticule over a flat block. */
export function MapSkeleton({ className }: { className?: string }) {
  return (
    <div aria-hidden="true" className={cn('relative h-full w-full overflow-hidden', className)}>
      <Skeleton className="absolute inset-0 rounded-none" />
      <div className="absolute inset-0 grid grid-cols-4 grid-rows-3">
        {Array.from({ length: 12 }, (_, index) => (
          <div key={index} className="border border-line/40" />
        ))}
      </div>
    </div>
  )
}

export function LoadingBlock({ label = 'Cargando…' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-xs text-fg-3">
      <Loader2 className="h-3.5 w-3.5 animate-spin" />
      {label}
    </div>
  )
}

interface StateProps {
  icon?: LucideIcon
  title: string
  description?: ReactNode
  action?: ReactNode
  className?: string
}

export function EmptyState({ icon: Icon = Inbox, title, description, action, className }: StateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center px-6 py-12 text-center', className)}>
      <Icon className="h-5 w-5 text-fg-3" />
      <p className="mt-3 text-sm font-medium text-fg-2">{title}</p>
      {description ? (
        <p className="mt-1 max-w-sm text-xs leading-relaxed text-fg-3">{description}</p>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  )
}

export function ErrorState({ title, description, action, className }: Omit<StateProps, 'icon'>) {
  return (
    <EmptyState
      icon={AlertTriangle}
      title={title}
      description={description}
      action={action}
      className={className}
    />
  )
}
