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

export async function apiGet<T>(
  path: string,
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  { signal, allowNotFound = false }: RequestOptions = {},
): Promise<T> {
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
    return null as T
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
  return parsed.data
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
