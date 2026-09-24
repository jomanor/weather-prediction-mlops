import maplibregl from 'maplibre-gl'
import { useEffect, useMemo, useRef, useState } from 'react'

import type { CurrentWeather } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { cn } from '@/lib/cn'
import { formatTemperature, isNum } from '@/lib/format'

import 'maplibre-gl/dist/maplibre-gl.css'

const STYLES = {
  dark: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  light: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
} as const

const IBERIA_CENTER: [number, number] = [-3.9, 39.9]

/* Geomatic layers. Free and keyless: AWS Open Data terrain tiles (Terrarium
 * encoding) for relief, RainViewer's public radar mosaic for precipitation. */
const DEM_SOURCE = 'meteoml-dem'
const HILLSHADE_LAYER = 'meteoml-hillshade'
const RADAR_SOURCE = 'meteoml-radar'
const RADAR_LAYER = 'meteoml-radar'
const RAINVIEWER_INDEX = 'https://api.rainviewer.com/public/weather-maps.json'

interface StationMapProps {
  stations: CurrentWeather[]
  selectedCity: string | null
  onSelect: (city: string) => void
  className?: string
}

export function StationMap({ stations, selectedCity, onSelect, className }: StationMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const markersRef = useRef(new globalThis.Map<string, maplibregl.Marker>())
  const radarTileRef = useRef<string | null>(null)
  const { isDark, units, mapLayers, setMapLayers } = usePreferences()
  const [radarReady, setRadarReady] = useState(true)

  const geoStations = useMemo(
    () => stations.filter((station) => isNum(station.latitude) && isNum(station.longitude)),
    [stations],
  )

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: isDark ? STYLES.dark : STYLES.light,
      center: IBERIA_CENTER,
      zoom: 5.1,
      attributionControl: { compact: true },
    })

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    mapRef.current = map

    // The container can settle after mount (lazy load, fonts, layout shifts),
    // which leaves the canvas short of its box. Track it explicitly.
    const observer = new ResizeObserver(() => map.resize())
    observer.observe(containerRef.current)
    map.once('load', () => map.resize())

    return () => {
      observer.disconnect()
      markersRef.current.forEach((marker) => marker.remove())
      markersRef.current.clear()
      map.remove()
      mapRef.current = null
    }
    // The map is created once; theme swaps are handled below via setStyle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    mapRef.current?.setStyle(isDark ? STYLES.dark : STYLES.light)
  }, [isDark])

  /* Sources and extra layers are wiped by setStyle; (re)apply idempotently. */
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const apply = () => {
      try {
        if (!map.getSource(DEM_SOURCE)) {
          map.addSource(DEM_SOURCE, {
            type: 'raster-dem',
            tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
            encoding: 'terrarium',
            tileSize: 256,
            maxzoom: 15,
            attribution: 'Relieve: AWS Terrain Tiles',
          })
        }
        if (!map.getLayer(HILLSHADE_LAYER)) {
          map.addLayer({
            id: HILLSHADE_LAYER,
            type: 'hillshade',
            source: DEM_SOURCE,
            paint: {
              'hillshade-exaggeration': 0.35,
              'hillshade-shadow-color': isDark ? '#000000' : '#5b6470',
            },
          })
        }
        map.setLayoutProperty(
          HILLSHADE_LAYER,
          'visibility',
          mapLayers.hillshade ? 'visible' : 'none',
        )

        if (mapLayers.terrain3d) {
          map.setTerrain({ source: DEM_SOURCE, exaggeration: 1.2 })
          if (map.getPitch() < 1) map.easeTo({ pitch: 55, duration: 600 })
        } else {
          map.setTerrain(null)
          if (map.getPitch() > 1) map.easeTo({ pitch: 0, duration: 600 })
        }

        if (mapLayers.radar) {
          void ensureRadarLayer(map, radarTileRef).then((ok) => setRadarReady(ok))
        } else if (map.getLayer(RADAR_LAYER)) {
          map.setLayoutProperty(RADAR_LAYER, 'visibility', 'none')
        }
      } catch {
        /* A failed tile source must not take the basemap down. */
      }
    }

    if (map.isStyleLoaded()) apply()
    else map.once('styledata', apply)
  }, [mapLayers, isDark])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const seen = new Set<string>()
    for (const station of geoStations) {
      const city = station.city
      seen.add(city)
      const selected = city === selectedCity
      const label = isNum(station.temperature)
        ? formatTemperature(station.temperature, units, 0)
        : '—'

      const existing = markersRef.current.get(city)
      const element = existing?.getElement()
      if (element) {
        element.dataset.selected = String(selected)
        element.innerHTML = renderPin(city, label)
      } else {
        const node = document.createElement('button')
        node.type = 'button'
        node.className = 'station-pin'
        node.dataset.selected = String(selected)
        node.innerHTML = renderPin(city, label)
        node.addEventListener('click', (event) => {
          event.stopPropagation()
          onSelect(city)
        })
        const marker = new maplibregl.Marker({ element: node, anchor: 'bottom' })
          .setLngLat([station.longitude as number, station.latitude as number])
          .addTo(map)
        markersRef.current.set(city, marker)
      }
    }

    for (const [city, marker] of markersRef.current) {
      if (!seen.has(city)) {
        marker.remove()
        markersRef.current.delete(city)
      }
    }
  }, [geoStations, selectedCity, units, onSelect])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !selectedCity) return
    const station = geoStations.find((candidate) => candidate.city === selectedCity)
    if (!station) return
    map.easeTo({
      center: [station.longitude as number, station.latitude as number],
      zoom: Math.max(map.getZoom(), 7),
      duration: 700,
    })
  }, [selectedCity, geoStations])

  const layerOptions = [
    { key: 'hillshade' as const, label: 'Relieve' },
    { key: 'radar' as const, label: 'Radar de lluvia' },
    { key: 'terrain3d' as const, label: 'Terreno 3D' },
  ]

  return (
    <div className={cn('relative h-full w-full', className)}>
      <div
        ref={containerRef}
        className="h-full w-full bg-panel-2"
        role="application"
        aria-label="Mapa de estaciones meteorológicas"
      />
      <div className="absolute right-2 top-2 z-10 w-40 rounded-[3px] border border-line bg-panel p-2.5">
        <div className="label mb-1.5">Capas</div>
        <div className="flex flex-col gap-1">
          {layerOptions.map((option) => (
            <label
              key={option.key}
              className="flex cursor-pointer items-center gap-2 text-xs text-fg-2 hover:text-fg"
            >
              <input
                type="checkbox"
                className="h-3.5 w-3.5 accent-[var(--accent)]"
                checked={mapLayers[option.key]}
                onChange={(event) =>
                  setMapLayers({ ...mapLayers, [option.key]: event.target.checked })
                }
              />
              {option.label}
            </label>
          ))}
        </div>
        {mapLayers.radar && !radarReady ? (
          <p className="mt-1.5 text-[10px] leading-tight text-warn">Radar no disponible ahora.</p>
        ) : null}
      </div>
    </div>
  )
}

/* Latest public RainViewer past frame; kept in a ref so the index is fetched
 * once per session. Returns false when the index or tiles are unavailable. */
async function ensureRadarLayer(
  map: maplibregl.Map,
  cache: { current: string | null },
): Promise<boolean> {
  if (!cache.current) {
    try {
      const response = await fetch(RAINVIEWER_INDEX, { signal: AbortSignal.timeout(8000) })
      if (!response.ok) return false
      const index: unknown = await response.json()
      const frames =
        (index as { radar?: { past?: Array<{ path?: string }> } })?.radar?.past ?? []
      const last = frames.at(-1)
      const host = (index as { host?: string })?.host
      if (!last?.path || !host) return false
      cache.current = `${host}${last.path}/256/{z}/{x}/{y}/2/1_1.png`
    } catch {
      return false
    }
  }
  if (!cache.current) return false

  if (!map.getSource(RADAR_SOURCE)) {
    map.addSource(RADAR_SOURCE, {
      type: 'raster',
      tiles: [cache.current],
      tileSize: 256,
      maxzoom: 10,
      attribution: 'Radar: RainViewer',
    })
  }
  if (!map.getLayer(RADAR_LAYER)) {
    map.addLayer({
      id: RADAR_LAYER,
      type: 'raster',
      source: RADAR_SOURCE,
      paint: { 'raster-opacity': 0.7 },
    })
  } else {
    map.setLayoutProperty(RADAR_LAYER, 'visibility', 'visible')
  }
  return true
}

function renderPin(city: string, temperature: string) {
  return `
    <span class="station-pin__dot" aria-hidden="true"></span>
    <span class="station-pin__body">
      <span class="station-pin__city">${escapeHtml(city)}</span>
      <span class="station-pin__temp">${escapeHtml(temperature)}</span>
    </span>
  `
}

function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, (character) => {
    switch (character) {
      case '&':
        return '&amp;'
      case '<':
        return '&lt;'
      case '>':
        return '&gt;'
      case '"':
        return '&quot;'
      default:
        return '&#39;'
    }
  })
}
