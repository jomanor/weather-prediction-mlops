import '@testing-library/jest-dom/vitest'

import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

afterEach(() => {
  cleanup()
})

/**
 * Web Storage for the jsdom environment.
 *
 * Node 22+ defines its own `localStorage`/`sessionStorage` global that only
 * works with `--localstorage-file`; it shadows the implementation jsdom
 * installs, so inside the vitest jsdom environment `globalThis.localStorage`
 * reads as `undefined` and anything seeding storage (`preferences.test.tsx`)
 * throws. Install an in-memory Storage whenever the environment's own is
 * missing or unusable — a no-op in a browser, where jsdom/Chrome provides it.
 */
class MemoryStorage implements Storage {
  #items = new Map<string, string>()

  get length(): number {
    return this.#items.size
  }

  clear(): void {
    this.#items.clear()
  }

  getItem(key: string): string | null {
    return this.#items.get(String(key)) ?? null
  }

  key(index: number): string | null {
    return [...this.#items.keys()][index] ?? null
  }

  removeItem(key: string): void {
    this.#items.delete(String(key))
  }

  setItem(key: string, value: string): void {
    this.#items.set(String(key), String(value))
  }
}

/** A storage object counts as usable when a real round-trip succeeds. */
function usable(candidate: unknown): candidate is Storage {
  if (!candidate || typeof (candidate as Storage).setItem !== 'function') return false
  const storage = candidate as Storage
  try {
    storage.setItem('__storage_probe__', '1')
    const stored = storage.getItem('__storage_probe__') === '1'
    storage.removeItem('__storage_probe__')
    return stored
  } catch {
    return false
  }
}

function installStorage(key: 'localStorage' | 'sessionStorage'): void {
  const scope = globalThis as unknown as Record<string, unknown>
  let current: unknown
  try {
    current = scope[key]
  } catch {
    current = undefined
  }
  if (usable(current)) return

  const storage = new MemoryStorage()
  Object.defineProperty(scope, key, { value: storage, writable: true, configurable: true })
  /* Vitest's jsdom global usually *is* `window`; mirror only when it is not. */
  if (typeof window !== 'undefined' && (window as unknown) !== (globalThis as unknown)) {
    Object.defineProperty(window, key, { value: storage, writable: true, configurable: true })
  }
}

installStorage('localStorage')
installStorage('sessionStorage')

// jsdom has no matchMedia; the preferences provider relies on it.
if (!window.matchMedia) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }))
}
