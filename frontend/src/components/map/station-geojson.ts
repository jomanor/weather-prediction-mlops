import type { Feature, FeatureCollection, Point } from 'geojson'
import type { GeoJSONSourceSpecification } from 'maplibre-gl'

import type { StationCollection, StationFeature } from '@/api/schemas'
import { weatherIconName } from '@/lib/weather'

export interface StationProperties {
  city: string
  temperature: number | null
  apparent_temperature: number | null
  relative_humidity: number | null
  wind_speed: number | null
  wind_direction: number | null
  precipitation: number | null
  weather_code: number | null
  observed_at: string
  /** Derived client-side: SDF image suffix (`wc-sun`, `wc-rain`, …). */
  weather_icon: string
}

export type StationFeatureCollection = FeatureCollection<Point, StationProperties>

/** One place for the clustering contract, shared by the map and its tests. */
export const STATION_CLUSTER_SOURCE = {
  cluster: true,
  clusterRadius: 44,
  clusterMaxZoom: 9,
} as const

export function stationSourceSpec(
  data: StationFeatureCollection,
): GeoJSONSourceSpecification {
  return { type: 'geojson', data, ...STATION_CLUSTER_SOURCE }
}

/**
 * Maps the `GET /api/map/stations` contract onto the GeoJSON the cluster source
 * consumes, adding the weather-icon key the symbol layer reads.
 */
export function toStationGeoJSON(collection: StationCollection | null): StationFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: (collection?.features ?? []).map(toFeature),
  }
}

function toFeature(feature: StationFeature): Feature<Point, StationProperties> {
  const properties = feature.properties
  return {
    type: 'Feature',
    geometry: { type: 'Point', coordinates: feature.geometry.coordinates },
    properties: {
      ...properties,
      weather_icon: weatherIconName(properties.weather_code),
    },
  }
}
