import type { ButtonHTMLAttributes, ReactNode, Ref } from 'react'

import { cn } from '@/lib/cn'

type Variant = 'primary' | 'outline' | 'ghost' | 'danger'
type Size = 'sm' | 'md' | 'icon'

const VARIANTS: Record<Variant, string> = {
  primary: 'border border-transparent bg-accent text-accent-fg hover:bg-accent/90 active:bg-accent/80',
  outline: 'border border-line bg-panel text-fg hover:border-line-strong hover:bg-panel-2',
  ghost: 'border border-transparent text-fg-2 hover:bg-panel-2 hover:text-fg',
  danger: 'border border-bad/40 text-bad hover:bg-bad/10',
}

/* Padding and gaps stay on the 4 px grid; heights are the control scale. */
const SIZES: Record<Size, string> = {
  sm: 'h-7 gap-1 px-3 text-body',
  md: 'h-9 gap-2 px-4 text-value',
  icon: 'h-8 w-8 justify-center',
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  children?: ReactNode
  /** React 19 passes `ref` as a prop; forwarded so callers can manage focus
   * (e.g. a dialog returning focus to the button that opened it). */
  ref?: Ref<HTMLButtonElement>
}

export function Button({
  variant = 'outline',
  size = 'md',
  className,
  type = 'button',
  ref,
  ...props
}: ButtonProps) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        'motion-ui inline-flex items-center rounded-control border font-medium',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
        'disabled:pointer-events-none disabled:opacity-45',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    />
  )
}
