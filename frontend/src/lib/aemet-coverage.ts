import type { City } from '@/api/schemas'

import { isNum } from '@/lib/format'

/**
 * AEMET OpenData only covers Spain, so a station on the American continent is
 * outside the benchmark's reference network — not an outage. The rule lives
 * here because three surfaces need it: the AEMET notice, the data-quality
 * meter and the Sistema city selectors.
 */
export type CoverageInput = Pick<City, 'longitude'>

/** True when the longitude places the station outside AEMET's coverage. */
export function isOutsideAemetCoverage(city?: CoverageInput): boolean {
  return Boolean(city && isNum(city.longitude) && city.longitude < -20)
}

/** Inverse of {@link isOutsideAemetCoverage}: the station is AEMET-covered. */
export function isAemetCovered(city?: CoverageInput): boolean {
  return !isOutsideAemetCoverage(city)
}

/** Registry entries inside AEMET coverage, in registry order. */
export function aemetCoveredCities<T extends CoverageInput>(cities: readonly T[]): T[] {
  return cities.filter(isAemetCovered)
}

/**
 * Names of the AEMET-covered stations. Rows from endpoints that carry no
 * coordinates (`/weather/quality`, `/analytics/diurnal`, `/benchmark`) are
 * matched against this set.
 */
export function aemetCoveredNames(cities: readonly (CoverageInput & { name: string })[]): Set<string> {
  return new Set(aemetCoveredCities(cities).map((city) => city.name))
}
