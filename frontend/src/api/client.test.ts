import { afterEach, describe, expect, it, vi } from 'vitest'
import { z } from 'zod'

import { ApiError, apiGet, apiGetWithDataAge, parseDataAgeSeconds } from '@/api/client'

/**
 * Transport contract for the freshness signal: `apiGet` keeps behaving exactly
 * as before (payload only) while `apiGetWithDataAge` additionally surfaces
 * `X-Data-Age-Seconds`, and never fails when the header is missing or junk.
 */

const schema = z.object({ count: z.number() })
const BODY = { count: 3 }

function jsonResponse(
  body: unknown = BODY,
  init: { status?: number; headers?: Record<string, string> } = {},
) {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { 'Content-Type': 'application/json', ...init.headers },
  })
}

function stubFetch(response: Response) {
  const fetchMock = vi.fn(async () => response)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('apiGet (unchanged contract)', () => {
  it('returns the parsed payload only', async () => {
    const fetchMock = stubFetch(jsonResponse(BODY, { headers: { 'X-Data-Age-Seconds': '90' } }))
    await expect(apiGet('/weather/current', schema)).resolves.toEqual(BODY)
    expect(fetchMock).toHaveBeenCalledWith('/api/weather/current', {
      signal: undefined,
      headers: { Accept: 'application/json' },
    })
  })

  it('still raises ApiError with the API detail on a non-ok response', async () => {
    stubFetch(jsonResponse({ detail: 'boom' }, { status: 500 }))
    await expect(apiGet('/weather/current', schema)).rejects.toBeInstanceOf(ApiError)
  })

  it('still returns null for a 404 when allowNotFound is set', async () => {
    stubFetch(jsonResponse({ detail: 'nope' }, { status: 404 }))
    await expect(apiGet('/weather/stats/x', schema, { allowNotFound: true })).resolves.toBeNull()
  })
})

describe('parseDataAgeSeconds', () => {
  it('accepts non-negative integers, with optional padding', () => {
    expect(parseDataAgeSeconds('0')).toBe(0)
    expect(parseDataAgeSeconds('120')).toBe(120)
    expect(parseDataAgeSeconds(' 42 ')).toBe(42)
  })

  it('rejects absent, negative, fractional and non-numeric values', () => {
    expect(parseDataAgeSeconds(null)).toBeNull()
    expect(parseDataAgeSeconds(undefined)).toBeNull()
    expect(parseDataAgeSeconds('')).toBeNull()
    expect(parseDataAgeSeconds('   ')).toBeNull()
    expect(parseDataAgeSeconds('-5')).toBeNull()
    expect(parseDataAgeSeconds('1.5')).toBeNull()
    expect(parseDataAgeSeconds('12s')).toBeNull()
    expect(parseDataAgeSeconds('abc')).toBeNull()
  })
})

describe('apiGetWithDataAge', () => {
  it('returns the payload and the server-reported age', async () => {
    stubFetch(jsonResponse(BODY, { headers: { 'X-Data-Age-Seconds': '1500' } }))
    await expect(apiGetWithDataAge('/weather/current', schema)).resolves.toEqual({
      data: BODY,
      dataAgeSeconds: 1500,
    })
  })

  it('reports null age when the header is absent (old backend, proxy)', async () => {
    stubFetch(jsonResponse(BODY))
    await expect(apiGetWithDataAge('/weather/current', schema)).resolves.toEqual({
      data: BODY,
      dataAgeSeconds: null,
    })
  })

  it('reports null age for junk values instead of throwing', async () => {
    for (const value of ['abc', '-5', '1.5', '']) {
      vi.unstubAllGlobals()
      stubFetch(jsonResponse(BODY, { headers: { 'X-Data-Age-Seconds': value } }))
      await expect(apiGetWithDataAge('/weather/current', schema)).resolves.toEqual({
        data: BODY,
        dataAgeSeconds: null,
      })
    }
  })

  it('keeps the 404 allowNotFound behaviour and has no age', async () => {
    stubFetch(jsonResponse({ detail: 'nope' }, { status: 404 }))
    await expect(
      apiGetWithDataAge('/weather/stats/x', schema, { allowNotFound: true }),
    ).resolves.toEqual({ data: null, dataAgeSeconds: null })
  })
})
