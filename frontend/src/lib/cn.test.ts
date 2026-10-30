import { describe, expect, it } from 'vitest'

import { cn } from '@/lib/cn'

describe('cn type scale', () => {
  it('keeps a custom size when a colour is also present', () => {
    expect(cn('text-label', 'text-fg-2')).toBe('text-label text-fg-2')
    expect(cn('text-body font-medium', 'text-fg')).toBe('text-body font-medium text-fg')
    expect(cn('text-page font-semibold', 'text-fg')).toBe('text-page font-semibold text-fg')
  })

  it('still resolves conflicting sizes and colours', () => {
    expect(cn('text-body', 'text-title')).toBe('text-title')
    expect(cn('text-fg-2', 'text-fg')).toBe('text-fg')
  })
})
