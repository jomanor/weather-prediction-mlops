import { describe, expect, it } from 'vitest'

/**
 * Guards the in-memory Storage installed by `setup.ts`. Node's native
 * `localStorage` global shadows jsdom's and is unusable without
 * `--localstorage-file`, which is what broke `preferences.test.tsx`.
 */
describe('test environment storage', () => {
  it('exposes a working localStorage on both globalThis and window', () => {
    localStorage.setItem('meteoml.probe', 'v')
    expect(localStorage.getItem('meteoml.probe')).toBe('v')
    expect(window.localStorage.getItem('meteoml.probe')).toBe('v')
    expect(localStorage.length).toBeGreaterThan(0)

    localStorage.removeItem('meteoml.probe')
    expect(localStorage.getItem('meteoml.probe')).toBeNull()
  })

  it('exposes sessionStorage and keeps the two stores separate', () => {
    sessionStorage.setItem('meteoml.probe', 's')
    expect(sessionStorage.getItem('meteoml.probe')).toBe('s')
    expect(localStorage.getItem('meteoml.probe')).toBeNull()

    sessionStorage.clear()
    expect(sessionStorage.length).toBe(0)
  })
})
