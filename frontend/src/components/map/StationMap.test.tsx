import { render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { StationCollection } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { StationMap } from '@/components/map/StationMap'

/* The interactive map is mocked at the module boundary: the assertions are the
   source spec and layer filters handed to MapLibre, not MapLibre itself. */
const mocks = vi.hoisted(() => ({
  addSource: vi.fn(),
  addLayer: vi.fn(),
}))

vi.mock('@/components/map/weather-icons', () => ({
  WEATHER_IMAGE_PREFIX: 'meteoml-wc-',
  WIND_ARROW_IMAGE: 'meteoml-wind-arrow',
  ensureMapIcons: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('maplibre-gl', () => {
  class FakeMap {
    layers = new Set<string>()
    sources = new Map<string, unknown>()
    constructor(_options: unknown) {}
    addControl() {}
    on() {}
    off() {}
    once(event: string, callback: () => void) {
      if (event === 'style.load') queueMicrotask(callback)
    }
    isStyleLoaded() {
      return false
    }
    getLayer(id: string) {
      return this.layers.has(id) ? {} : undefined
    }
    getSource(id: string) {
      return this.sources.get(id)
    }
    addSource(id: string, spec: unknown) {
      mocks.addSource(id, spec)
      this.sources.set(id, {})
    }
    addLayer(layer: { id: string }) {
      mocks.addLayer(layer)
      this.layers.add(layer.id)
    }
    setLayoutProperty() {}
    setTerrain() {}
    easeTo() {}
    getPitch() {
      return 0
    }
    getZoom() {
      return 5
    }
    queryRenderedFeatures() {
      return []
    }
    getCanvas() {
      return { style: {}, setAttribute: () => {} }
    }
    resize() {}
    remove() {}
    setStyle() {}
    setFilter() {}
    hasImage() {
      return false
    }
    addImage() {}
  }
  class FakeNavigationControl {
    constructor(_options?: unknown) {}
  }
  return {
    default: { Map: FakeMap, NavigationControl: FakeNavigationControl },
    Map: FakeMap,
    NavigationControl: FakeNavigationControl,
  }
})

const COLLECTION: StationCollection = {
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
        weather_code: 0,
        observed_at: '2026-09-25T11:00:00Z',
      },
    },
  ],
}

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeEach(() => {
  mocks.addSource.mockClear()
  mocks.addLayer.mockClear()
  vi.stubGlobal('ResizeObserver', ResizeObserverStub)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function renderMap() {
  return render(
    <PreferencesProvider>
      <StationMap collection={COLLECTION} selectedCity={null} onSelect={vi.fn()} />
    </PreferencesProvider>,
  )
}

describe('StationMap cluster source (U5)', () => {
  it('adds the stations as a clustered GeoJSON source', async () => {
    renderMap()
    await waitFor(() => expect(mocks.addSource).toHaveBeenCalled())

    const call = mocks.addSource.mock.calls.find(([id]) => id === 'meteoml-stations')
    expect(call).toBeTruthy()
    const spec = call?.[1] as Record<string, unknown>
    expect(spec.type).toBe('geojson')
    expect(spec.cluster).toBe(true)
    expect(spec.clusterRadius).toBe(44)
    expect(spec.clusterMaxZoom).toBe(9)
  })

  it('excludes present-but-null wind from the arrow layer filter', async () => {
    renderMap()
    await waitFor(() => expect(mocks.addLayer).toHaveBeenCalled())

    const wind = mocks.addLayer.mock.calls
      .map(([layer]) => layer as { id: string; filter?: unknown })
      .find((layer) => layer.id === 'meteoml-stations-wind')
    expect(wind).toBeTruthy()
    const serialised = JSON.stringify(wind?.filter)
    // `['has', …]` is true for a null property; the filter must compare values.
    expect(serialised).toContain('["!=",["get","wind_speed"],null]')
    expect(serialised).toContain('["!=",["get","wind_direction"],null]')
    expect(serialised).not.toContain('"has","wind_speed"')
  })
})
