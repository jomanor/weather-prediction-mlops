import type { z } from 'zod'

/**
 * Base URL for the FastAPI service. Defaults to the relative `/api`, which is
 * proxied by Vite in development and by the Netlify edge redirect in
 * production, so the app never needs a hard-coded host.
 */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/+$/, '')

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(message: string, status: number, detail?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }

  get isNotFound() {
    return this.status === 404
  }
}

export interface RequestOptions {
  signal?: AbortSignal
  /** Treat a 404 as `null` instead of throwing. Useful for optional series. */
  allowNotFound?: boolean
}

export interface WriteOptions extends RequestOptions {
  body?: unknown
}

/** POST/DELETE/PUT against the API, parsing the (optional) response body. */
export async function apiSend<T>(
  method: 'POST' | 'DELETE' | 'PUT',
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown> | null,
  { signal, body, allowNotFound = false }: WriteOptions = {},
): Promise<T | null> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      signal,
      headers: {
        Accept: 'application/json',
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new ApiError('No se pudo contactar con la API.', 0, cause)
  }

  if (response.status === 404 && allowNotFound) return null
  if (!response.ok) {
    const detail = await readDetail(response)
    throw new ApiError(detail ?? `La API respondió ${response.status}`, response.status, detail)
  }
  if (!schema) return null

  const payload: unknown = await response.json()
  const parsed = schema.safeParse(payload)
  if (!parsed.success) {
    const issue = parsed.error.issues[0]
    throw new ApiError(
      `Respuesta inesperada de la API en ${path} (${issue?.path.join('.')}: ${issue?.message})`,
      0,
      parsed.error.issues,
    )
  }
  return parsed.data
}

/**
 * One GET + schema validation, keeping the `Response` so a caller can read
 * headers. Shared by `apiGet` (payload only) and `apiGetWithDataAge`.
 */
async function getJson<T>(
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  { signal, allowNotFound = false }: RequestOptions = {},
): Promise<{ data: T; response: Response }> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      signal,
      headers: { Accept: 'application/json' },
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new ApiError('No se pudo contactar con la API.', 0, cause)
  }

  if (response.status === 404 && allowNotFound) {
    return { data: null as T, response }
  }

  if (!response.ok) {
    const detail = await readDetail(response)
    throw new ApiError(detail ?? `La API respondió ${response.status}`, response.status, detail)
  }

  const payload: unknown = await response.json()
  const parsed = schema.safeParse(payload)
  if (!parsed.success) {
    const issue = parsed.error.issues[0]
    throw new ApiError(
      `Respuesta inesperada de la API en ${path} (${issue?.path.join('.')}: ${issue?.message})`,
      0,
      parsed.error.issues,
    )
  }
  return { data: parsed.data, response }
}

export async function apiGet<T>(
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  options: RequestOptions = {},
): Promise<T> {
  const { data } = await getJson(path, schema, options)
  return data
}

/** Response header the API sets on the endpoints that know the data's age. */
export const DATA_AGE_HEADER = 'X-Data-Age-Seconds'

export interface DataAgeResponse<T> {
  data: T
  /**
   * Age in whole seconds of the freshest observation at response time, or
   * `null` when the header is absent (older backend, stripped by a proxy) or
   * does not parse to a non-negative integer.
   *
   * Lower bound only, never live authority: `/weather/current` is served
   * `public, max-age=30` with a strong ETag, so a revalidation answers 304 and
   * the browser replays the cached body *and* this stored header on a response
   * that looks like a normal 200 (a 304 never reaches `fetch`). Consumers must
   * therefore treat it as a floor — see `effectiveAgeSeconds`, which keeps the
   * larger of this carried-forward value and the payload's own timestamps.
   */
  dataAgeSeconds: number | null
}

/** Strictly a non-negative integer: `'12.5'`, `'-3'`, `'abc'`, `''` all yield null. */
export function parseDataAgeSeconds(value: string | null | undefined): number | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) return null
  const parsed = Number(trimmed)
  return Number.isSafeInteger(parsed) ? parsed : null
}

/**
 * `apiGet` plus the server-computed data age, for the surfaces that render a
 * freshness signal. Additive: every other call site keeps calling `apiGet`.
 */
export async function apiGetWithDataAge<T>(
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  options: RequestOptions = {},
): Promise<DataAgeResponse<T>> {
  const { data, response } = await getJson(path, schema, options)
  return {
    data,
    dataAgeSeconds: parseDataAgeSeconds(response.headers.get(DATA_AGE_HEADER)),
  }
}

async function readDetail(response: Response): Promise<string | null> {
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      return typeof detail === 'string' ? detail : null
    }
  } catch {
    /* non-JSON error body */
  }
  return null
}
