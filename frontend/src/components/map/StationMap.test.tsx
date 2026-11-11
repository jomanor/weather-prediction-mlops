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

/* Finding 16: one elevation setting. The 3D terrain mesh and the building
   extrusions are gone, so there is one DEM source and nothing for the hillshade
   to cancel against. */
describe('StationMap elevation', () => {
  it('adds exactly one DEM source and no 3D layer', async () => {
    renderMap()
    await waitFor(() => expect(mocks.addLayer).toHaveBeenCalled())

    const demSources = mocks.addSource.mock.calls
      .map(([id, spec]) => ({ id, type: (spec as { type?: string }).type }))
      .filter((source) => source.type === 'raster-dem')
    expect(demSources.map((source) => source.id)).toEqual(['meteoml-dem-hillshade'])

    const layerIds = mocks.addLayer.mock.calls.map(([layer]) => (layer as { id: string }).id)
    expect(layerIds).toContain('meteoml-hillshade')
    expect(layerIds.filter((id) => /buildings|terrain/.test(id))).toEqual([])
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
     them through "Capas" and a toggle leaves the sheet open (finding 16). */
  beforeEach(() => localStorage.clear())
  afterEach(() => localStorage.clear())

  it('keeps the sheet open while a layer is toggled, and persists the toggle', async () => {
    const user = userEvent.setup()
    renderMap()

    /* Every query is scoped to the sheet: the desktop panel is still mounted
       (CSS hides it, jsdom does not resolve the media query). */
    const trigger = () => screen.getByRole('button', { name: /^capas$/i })
    const sheet = () => within(screen.getByRole('dialog', { name: 'Capas del mapa' }))

    await user.click(trigger())
    await user.click(sheet().getByLabelText('Relieve'))

    /* Both toggles stay reachable: the sheet does not close on a change. */
    expect(sheet().getByLabelText('Relieve')).toBeChecked()
    expect(sheet().getByLabelText('Radar de lluvia')).toBeInTheDocument()
    expect(screen.getByRole('dialog', { name: 'Capas del mapa' })).toBeInTheDocument()

    await user.click(sheet().getByLabelText('Radar de lluvia'))
    expect(sheet().getByLabelText('Radar de lluvia')).toBeChecked()
    /* One elevation setting: turning radar on cannot uncheck relief. */
    expect(sheet().getByLabelText('Relieve')).toBeChecked()
    expect(JSON.parse(localStorage.getItem('meteoml.map') ?? '{}')).toEqual({
      hillshade: true,
      radar: true,
    })
  })

  it('renders the desktop panel and the phone trigger from one set of controls', () => {
    renderMap()

    expect(screen.getByRole('button', { name: /^capas$/i })).toBeInTheDocument()
    /* Closed sheet: only the desktop panel's content is mounted. */
    expect(screen.getAllByLabelText('Radar de lluvia')).toHaveLength(1)
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
