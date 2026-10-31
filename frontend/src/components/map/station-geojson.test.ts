import { describe, expect, it } from 'vitest'

import { toStationGeoJSON, stationSourceSpec } from '@/components/map/station-geojson'
import type { StationCollection } from '@/api/schemas'

function collection(
  weatherCode: number | null,
  isDay: 0 | 1 | null = null,
): StationCollection {
  return {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [-3.7, 40.4] },
        properties: {
          city: 'Madrid',
          temperature: 24.3,
          apparent_temperature: null,
          relative_humidity: null,
          wind_speed: 12,
          wind_direction: 210,
          precipitation: 0,
          weather_code: weatherCode,
          is_day: isDay,
          observed_at: '2026-09-25T11:00:00Z',
        },
      },
    ],
  }
}

describe('toStationGeoJSON', () => {
  it('preserves geometry and adds the weather icon key', () => {
    /* Pass the flag explicitly: a null is_day falls back to the runner's wall clock. */
    const result = toStationGeoJSON(collection(0, 1))
    expect(result.features[0].geometry.coordinates).toEqual([-3.7, 40.4])
    expect(result.features[0].properties.weather_icon).toBe('sun')
    expect(result.features[0].properties.temperature).toBe(24.3)
  })

  it('maps day/night codes onto sun or moon glyphs per station', () => {
    expect(toStationGeoJSON(collection(0, 1)).features[0].properties.weather_icon).toBe('sun')
    expect(toStationGeoJSON(collection(0, 0)).features[0].properties.weather_icon).toBe('moon')
    expect(toStationGeoJSON(collection(1, 0)).features[0].properties.weather_icon).toBe('moon-star')
    expect(toStationGeoJSON(collection(2, 0)).features[0].properties.weather_icon).toBe('cloud-moon')
    expect(toStationGeoJSON(collection(3, 0)).features[0].properties.weather_icon).toBe('cloud-moon')
    expect(toStationGeoJSON(collection(2, 1)).features[0].properties.weather_icon).toBe('cloud-sun')
    /* Night has no variant for these codes: the rain/snow/fog glyph stays. */
    expect(toStationGeoJSON(collection(61, 0)).features[0].properties.weather_icon).toBe('rain')
  })

  it('carries the per-station is_day flag through to the properties', () => {
    expect(toStationGeoJSON(collection(0, 0)).features[0].properties.is_day).toBe(0)
    expect(toStationGeoJSON(collection(0, null)).features[0].properties.is_day).toBeNull()
  })

  it('maps storm / snow / fog codes onto their glyphs', () => {
    expect(toStationGeoJSON(collection(95)).features[0].properties.weather_icon).toBe('storm')
    expect(toStationGeoJSON(collection(73)).features[0].properties.weather_icon).toBe('snow')
    expect(toStationGeoJSON(collection(45)).features[0].properties.weather_icon).toBe('fog')
  })

  it('falls back to cloud for an unknown or missing code', () => {
    expect(toStationGeoJSON(collection(999)).features[0].properties.weather_icon).toBe('cloud')
    expect(toStationGeoJSON(collection(null)).features[0].properties.weather_icon).toBe('cloud')
  })

  it('handles a missing collection', () => {
    expect(toStationGeoJSON(null).features).toEqual([])
  })
})

describe('stationSourceSpec', () => {
  it('builds a clustered GeoJSON source with the shared radius', () => {
    const spec = stationSourceSpec(toStationGeoJSON(collection(0)))
    expect(spec.type).toBe('geojson')
    expect(spec.cluster).toBe(true)
    expect(spec.clusterRadius).toBe(44)
    expect(spec.clusterMaxZoom).toBe(9)
  })
})
