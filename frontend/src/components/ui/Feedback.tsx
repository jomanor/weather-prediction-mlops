import { AlertTriangle, Inbox, Loader2, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded bg-panel-2', className)} />
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
