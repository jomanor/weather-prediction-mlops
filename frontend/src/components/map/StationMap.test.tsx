import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { StationCollection } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { StationMap } from '@/components/map/StationMap'

/* The interactive map is mocked at the module boundary: the assertions are the
   source spec and layer filters handed to MapLibre, not MapLibre itself. */
const mocks = vi.hoisted(() => ({
  addSource: vi.fn(),
  addLayer: vi.fn(),
  /* Toggled per test: false models the raster relief basemap (no vector tiles). */
  vectorBasemap: { value: true },
}))

vi.mock('@/components/map/weather-icons', () => ({
  WEATHER_IMAGE_PREFIX: 'meteoml-wc-',
  WIND_ARROW_IMAGE: 'meteoml-wind-arrow',
  ensureMapIcons: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('maplibre-gl', () => {
  class FakeMap {
    layers = new Set<string>()
    /* The Carto styles the app ships expose their vector tiles as `protomaps`;
       the relief basemap has no vector source at all. */
    sources = new Map<string, unknown>(
      mocks.vectorBasemap.value ? [['protomaps', {}]] : [],
    )
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
    addLayer(layer: { id: string }, beforeId?: string) {
      mocks.addLayer(layer, beforeId)
      this.layers.add(layer.id)
    }
    getStyle() {
      return { layers: [{ id: 'watername', type: 'symbol' }] }
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
        is_day: 1,
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

describe('StationMap 3D buildings', () => {
  it('extrudes the Carto buildings source-layer with its height field', async () => {
    renderMap()
    await waitFor(() => expect(mocks.addLayer).toHaveBeenCalled())

    const call = mocks.addLayer.mock.calls.find(
      ([layer]) => (layer as { id?: string }).id === 'meteoml-buildings-3d',
    )
    expect(call).toBeTruthy()
    const [layer, beforeId] = call as [Record<string, unknown>, string | undefined]
    expect(layer.source).toBe('protomaps')
    expect(layer['source-layer']).toBe('buildings')
    expect(layer.minzoom).toBe(14)
    expect((layer.paint as Record<string, unknown>)['fill-extrusion-height']).toEqual([
      'coalesce',
      ['get', 'height'],
      ['get', 'render_height'],
      6,
    ])
    /* Inserted under the basemap labels, not on top of them. */
    expect(beforeId).toBe('watername')
  })

  it('leaves the raster relief basemap without a 3D layer', async () => {
    mocks.vectorBasemap.value = false
    try {
      renderMap()
      await waitFor(() => expect(mocks.addSource).toHaveBeenCalled())
      expect(
        mocks.addLayer.mock.calls.some(
          ([layer]) => (layer as { id?: string }).id === 'meteoml-buildings-3d',
        ),
      ).toBe(false)
    } finally {
      mocks.vectorBasemap.value = true
    }
  })
})

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

describe('StationMap layer controls on a phone', () => {
  /* The sheet and the desktop panel are the same controls; the phone reaches
     them through "Capas" and the map still owns the DEM/3D exclusivity rule. */
  beforeEach(() => localStorage.clear())
  afterEach(() => localStorage.clear())

  it('keeps relief and 3D terrain mutually exclusive from the sheet', async () => {
    const user = userEvent.setup()
    renderMap()

    /* Every query is scoped to the sheet: the desktop panel is still mounted
       (CSS hides it, jsdom does not resolve the media query). */
    const trigger = () => screen.getByRole('button', { name: /^capas$/i })
    const sheet = () => within(screen.getByRole('dialog', { name: 'Capas del mapa' }))

    await user.click(trigger())
    await user.click(sheet().getByLabelText('Terreno 3D'))
    /* A selection closes the sheet. */
    expect(screen.queryByRole('dialog')).toBeNull()

    await user.click(trigger())
    expect(sheet().getByLabelText('Terreno 3D')).toBeChecked()
    expect(sheet().getByLabelText('Relieve')).not.toBeChecked()

    await user.click(sheet().getByLabelText('Relieve'))
    expect(screen.queryByRole('dialog')).toBeNull()

    await user.click(trigger())
    expect(sheet().getByLabelText('Relieve')).toBeChecked()
    expect(sheet().getByLabelText('Terreno 3D')).not.toBeChecked()
  })

  it('renders the desktop panel and the phone trigger from one set of controls', () => {
    renderMap()

    expect(screen.getByRole('button', { name: /^capas$/i })).toBeInTheDocument()
    /* Closed sheet: only the desktop panel's content is mounted. */
    expect(screen.getAllByLabelText('Edificios 3D')).toHaveLength(1)
    expect(screen.getAllByLabelText('Base')).toHaveLength(1)
  })
})

describe('StationMap accessible name', () => {
  it('names the map as a region instead of an inert aria-label on a div', () => {
    renderMap()
    /* A plain <div aria-label> is exposed to nobody; the region role makes the
       name reachable in the landmark list. */
    const map = screen.getByRole('region', { name: 'Mapa de estaciones meteorológicas' })
    expect(map).toBeInTheDocument()
  })
})
