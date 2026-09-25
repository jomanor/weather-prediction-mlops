import { render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Skeleton } from '@/components/ui/Feedback'

function setReducedMotion(reduce: boolean) {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    () =>
      ({
        matches: reduce,
        media: '(prefers-reduced-motion: reduce)',
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList,
  )
}

afterEach(() => vi.restoreAllMocks())

describe('Skeleton reduced-motion gating', () => {
  it('shimmers when motion is allowed', () => {
    setReducedMotion(false)
    const { container } = render(<Skeleton className="h-3" />)
    expect(container.firstElementChild).toHaveClass('shimmer')
  })

  it('renders a static block when reduced motion is requested', () => {
    setReducedMotion(true)
    const { container } = render(<Skeleton className="h-3" />)
    expect(container.firstElementChild).not.toHaveClass('shimmer')
  })
})
