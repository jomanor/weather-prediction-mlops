import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { usePrefersReducedMotion } from '@/hooks/usePrefersReducedMotion'

type ChangeListener = (event: { matches: boolean }) => void

function mockMatchMedia(matches: boolean) {
  const listeners = new Set<ChangeListener>()
  const media = {
    matches,
    media: '(prefers-reduced-motion: reduce)',
    onchange: null,
    addEventListener: (_type: string, listener: ChangeListener) => listeners.add(listener),
    removeEventListener: (_type: string, listener: ChangeListener) => listeners.delete(listener),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }
  vi.spyOn(window, 'matchMedia').mockImplementation(
    () => media as unknown as MediaQueryList,
  )
  return { listeners }
}

afterEach(() => vi.restoreAllMocks())

describe('usePrefersReducedMotion', () => {
  it('reflects the current OS preference', () => {
    mockMatchMedia(true)
    const { result } = renderHook(() => usePrefersReducedMotion())
    expect(result.current).toBe(true)
  })

  it('updates on a live change and unsubscribes on unmount', () => {
    const { listeners } = mockMatchMedia(false)
    const { result, unmount } = renderHook(() => usePrefersReducedMotion())
    expect(result.current).toBe(false)

    act(() => listeners.forEach((listener) => listener({ matches: true })))
    expect(result.current).toBe(true)

    unmount()
    expect(listeners.size).toBe(0)
  })
})
