import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

type Tone = 'neutral' | 'accent' | 'ok' | 'warn' | 'bad' | 'model' | 'aemet' | 'obs'

const TONES: Record<Tone, string> = {
  neutral: 'border-line text-fg-2',
  accent: 'border-accent/35 text-accent bg-accent-soft',
  ok: 'border-ok/35 text-ok',
  warn: 'border-warn/35 text-warn',
  bad: 'border-bad/35 text-bad',
  model: 'border-model/40 text-model',
  aemet: 'border-aemet/40 text-aemet',
  obs: 'border-line-strong text-fg',
}

/**
 * A short status chip. Tone is decorative reinforcement only — the label always
 * carries the meaning (never colour alone).
 */
export function Badge({
  children,
  tone = 'neutral',
  className,
  title,
}: {
  children: ReactNode
  tone?: Tone
  className?: string
  title?: string
}) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex items-center gap-1 rounded-control border px-2 py-1 font-mono text-badge leading-none font-medium uppercase tracking-wider',
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

export function StatusDot({
  tone = 'neutral',
  pulse = false,
  className,
}: {
  tone?: Tone
  pulse?: boolean
  className?: string
}) {
  const colors: Record<Tone, string> = {
    neutral: 'bg-fg-3',
    accent: 'bg-accent',
    ok: 'bg-ok',
    warn: 'bg-warn',
    bad: 'bg-bad',
    model: 'bg-model',
    aemet: 'bg-aemet',
    obs: 'bg-fg',
  }
  return (
    <span className={cn('relative inline-flex h-1.5 w-1.5', className)} aria-hidden="true">
      {pulse ? (
        <span
          className={cn('absolute inline-flex h-full w-full animate-ping rounded-full opacity-60', colors[tone])}
        />
      ) : null}
      <span className={cn('relative inline-flex h-1.5 w-1.5 rounded-full', colors[tone])} />
    </span>
  )
}
