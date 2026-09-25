import { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

/**
 * Typed search-param state for react-router v7.
 *
 * The schema must be a stable (module-scope) object. Values are written in the
 * schema's declaration order so URLs are deterministic, params equal to their
 * default are omitted, and unknown params owned by other surfaces survive.
 * Parsing never throws: malformed or unknown values resolve to the default.
 */

/** Codec for a single search param. */
export interface UrlCodec<T> {
  readonly fallback: T
  parse(raw: string | null): T
  format(value: T): string
}

export type UrlSchema = Record<string, UrlCodec<unknown>>

export type UrlState<S extends UrlSchema> = {
  [K in keyof S]: S[K] extends UrlCodec<infer V> ? V : never
}

export type UrlPatch<S extends UrlSchema> = Partial<UrlState<S>>

/** Free-form string param; absent values resolve to the fallback. */
export const urlString = (fallback = ''): UrlCodec<string> => ({
  fallback,
  parse: (raw) => raw ?? fallback,
  format: (value) => value,
})

/** Enum param over a fixed set of string/number options; anything else falls back. */
export const urlOption = <T extends string | number>(
  values: readonly T[],
  fallback: T,
): UrlCodec<T> => ({
  fallback,
  parse: (raw) => values.find((value) => String(value) === raw) ?? fallback,
  format: (value) => String(value),
})

function parseUrlState(params: URLSearchParams, schema: UrlSchema): Record<string, unknown> {
  const state: Record<string, unknown> = {}
  for (const key of Object.keys(schema)) state[key] = schema[key].parse(params.get(key))
  return state
}

function applyUrlPatch(
  prev: URLSearchParams,
  patch: Record<string, unknown>,
  schema: UrlSchema,
): URLSearchParams {
  const merged = { ...parseUrlState(prev, schema), ...patch }
  const next = new URLSearchParams()

  for (const key of Object.keys(schema)) {
    const codec = schema[key]
    const value = merged[key]
    if (value === undefined || value === codec.fallback) continue
    next.set(key, codec.format(value))
  }
  /* Params this schema does not own are preserved in their original order. */
  for (const [key, value] of prev.entries()) {
    if (!(key in schema)) next.append(key, value)
  }
  return next
}

export function useUrlState<S extends UrlSchema>(
  schema: S,
  options: { replace?: boolean } = {},
): [UrlState<S>, (patch: UrlPatch<S>, options?: { replace?: boolean }) => void] {
  const [searchParams, setSearchParams] = useSearchParams()
  const defaultReplace = options.replace ?? true
  const search = searchParams.toString()

  const state = useMemo(
    () => parseUrlState(new URLSearchParams(search), schema) as UrlState<S>,
    [search, schema],
  )

  const setState = useCallback(
    (patch: UrlPatch<S>, setOptions?: { replace?: boolean }) => {
      setSearchParams((prev) => applyUrlPatch(prev, patch as Record<string, unknown>, schema), {
        /* `replace` by default so scrubbing/dragging cannot flood the back stack. */
        replace: setOptions?.replace ?? defaultReplace,
      })
    },
    [setSearchParams, schema, defaultReplace],
  )

  return [state, setState]
}
