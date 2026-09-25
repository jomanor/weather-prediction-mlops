import { act, renderHook } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { urlOption, urlString, useUrlState, type UrlCodec } from '@/hooks/useUrlState'

type Hours = 24 | 48 | 72
type Tab = 'red' | 'estaciones'

const SCHEMA: { city: UrlCodec<string>; hours: UrlCodec<Hours>; tab: UrlCodec<Tab> } = {
  city: urlString(''),
  hours: urlOption([24, 48, 72] as const, 24),
  tab: urlOption(['red', 'estaciones'] as const, 'red'),
}

function useProbe() {
  const [state, setState] = useUrlState(SCHEMA)
  const location = useLocation()
  return { state, setState, search: location.search }
}

function renderProbe(entry: string) {
  return renderHook(() => useProbe(), {
    wrapper: ({ children }) => <MemoryRouter initialEntries={[entry]}>{children}</MemoryRouter>,
  })
}

describe('useUrlState', () => {
  it('returns the defaults for an empty URL', () => {
    const { result } = renderProbe('/')
    expect(result.current.state).toEqual({ city: '', hours: 24, tab: 'red' })
    expect(result.current.search).toBe('')
  })

  it('round-trips a patch through the search params', () => {
    const { result } = renderProbe('/')
    act(() => result.current.setState({ hours: 48, city: 'Madrid' }))
    expect(result.current.state.hours).toBe(48)
    expect(result.current.state.city).toBe('Madrid')
    expect(result.current.search).toBe('?city=Madrid&hours=48')
  })

  it('falls back to the default for malformed or unknown values', () => {
    const { result } = renderProbe('/?hours=abc&tab=nope&city=')
    expect(result.current.state).toEqual({ city: '', hours: 24, tab: 'red' })
  })

  it('keeps declaration order and preserves foreign params', () => {
    const { result } = renderProbe('/?ref=abc')
    act(() => result.current.setState({ tab: 'estaciones' }))
    act(() => result.current.setState({ city: 'Sevilla', hours: 72 }))
    expect(result.current.search).toBe('?city=Sevilla&hours=72&tab=estaciones&ref=abc')
  })

  it('omits params that equal their default', () => {
    const { result } = renderProbe('/?city=Madrid&hours=48&tab=estaciones')
    act(() => result.current.setState({ hours: 24, tab: 'red', city: '' }))
    expect(result.current.search).toBe('')
    expect(result.current.state).toEqual({ city: '', hours: 24, tab: 'red' })
  })
})
