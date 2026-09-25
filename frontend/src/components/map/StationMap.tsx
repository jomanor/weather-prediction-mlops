import maplibregl from 'maplibre-gl'
import type {
  FilterSpecification,
  GeoJSONSource,
  MapMouseEvent,
  PropertyValueSpecification,
  RasterTileSource,
  StyleSpecification,
} from 'maplibre-gl'
import { useEffect, useMemo, useRef } from 'react'

import type { StationCollection } from '@/api/schemas'
import { usePreferences, type BasemapId } from '@/app/preferences'
import { MAP_PAINT } from '@/lib/chart-theme'
import { cn } from '@/lib/cn'
import { RAMP_DOMAINS, rampBreaks, rampHex, rampToken, type RampName } from '@/lib/ramps'

import { RadarControls } from './RadarControls'
import { radarTileUrl, useRadar } from './radar'
import { stationSourceSpec, toStationGeoJSON } from './station-geojson'
import { WEATHER_IMAGE_PREFIX, WIND_ARROW_IMAGE, ensureMapIcons } from './weather-icons'

import 'maplibre-gl/dist/maplibre-gl.css'

/* Keyless providers only. Carto supplies the vector basemaps; OpenTopoMap the
 * raster relief basemap (it has no vector `carto` source, so 3D buildings are
 * unavailable there — guarded below). */
const CARTO_STYLES: Record<'positron' | 'dark', string> = {
  positron: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
  dark: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
}

const TOPO_STYLE: StyleSpecification = {
  version: 8,
  // Symbol layers (cluster counts) need a glyph source; this one is keyless.
  glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
  sources: {
    opentopomap: {
      type: 'raster',
      tiles: ['https://tile.opentopomap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      maxzoom: 17,
      attribution: '© OpenStreetMap · SRTM · © OpenTopoMap',
    },
  },
  layers: [{ id: 'opentopomap', type: 'raster', source: 'opentopomap' }],
}

export function basemapStyle(basemap: BasemapId): string | StyleSpecification {
  return basemap === 'topo' ? TOPO_STYLE : CARTO_STYLES[basemap]
}

const IBERIA_CENTER: [number, number] = [-3.9, 39.9]

const DEM_SOURCE = 'meteoml-dem-hillshade'
const TERRAIN_DEM_SOURCE = 'meteoml-dem-terrain'
const HILLSHADE_LAYER = 'meteoml-hillshade'
const BUILDINGS_LAYER = 'meteoml-buildings-3d'
const RADAR_SOURCE = 'meteoml-radar'
const RADAR_LAYER = 'meteoml-radar'
const STATIONS_SOURCE = 'meteoml-stations'
const STATION_HALO_LAYER = 'meteoml-stations-halo'
const STATION_SELECTED_LAYER = 'meteoml-stations-selected'
const STATION_WIND_LAYER = 'meteoml-stations-wind'
const STATION_POINTS_LAYER = 'meteoml-stations-points'
const CLUSTERS_LAYER = 'meteoml-stations-clusters'
const CLUSTER_COUNT_LAYER = 'meteoml-stations-cluster-count'
const FLAT_BUILDING_LAYERS = ['building', 'building-top']
const UNCLUSTERED_FILTER: FilterSpecification = ['!', ['has', 'point_count']]

/** Map transitions follow the OS motion preference. */
function motionDuration(ms: number): number {
  return typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ? 0
    : ms
}

/** Data-driven `['step', …]` expression straight from the token ramp. */
function rampStepExpression(name: RampName, property: string): PropertyValueSpecification<string> {
  const values: Array<string | number> = [rampHex(name, 0)]
  rampBreaks(name, RAMP_DOMAINS[name]).forEach((threshold, index) => {
    values.push(threshold, rampHex(name, index + 1))
  })
  return ['step', ['get', property], ...values] as unknown as PropertyValueSpecification<string>
}

interface StationMapProps {
  collection: StationCollection | null
  selectedCity: string | null
  onSelect: (city: string) => void
  className?: string
}

export function StationMap({ collection, selectedCity, onSelect, className }: StationMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const { isDark, palette, mapLayers, setMapLayers, basemap, setBasemap } = usePreferences()
  const radar = useRadar(mapLayers.radar)

  const geoJson = useMemo(() => toStationGeoJSON(collection), [collection])

  /* Latest props/state for callbacks bound once at map creation. */
  const onSelectRef = useRef(onSelect)
  const selectedCityRef = useRef(selectedCity)
  const geoJsonRef = useRef(geoJson)
  const basemapRef = useRef(basemap)
  const radarUrlRef = useRef<string | null>(null)

  useEffect(() => {
    onSelectRef.current = onSelect
  }, [onSelect])
  useEffect(() => {
    selectedCityRef.current = selectedCity
  }, [selectedCity])
  useEffect(() => {
    geoJsonRef.current = geoJson
  }, [geoJson])
  useEffect(() => {
    basemapRef.current = basemap
  }, [basemap])
  radarUrlRef.current =
    radar.host && radar.current ? radarTileUrl(radar.host, radar.current) : null

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: basemapStyle(basemapRef.current),
      center: IBERIA_CENTER,
      zoom: 5.1,
      attributionControl: { compact: true },
    })

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    mapRef.current = map

    /* One delegated handler per event; layer ids are checked at event time so
       it survives style swaps and StrictMode remounts. */
    const handleClick = (event: MapMouseEvent) => {
      const clusterLayers = [CLUSTERS_LAYER].filter((id) => map.getLayer(id))
      if (clusterLayers.length) {
        const feature = map.queryRenderedFeatures(event.point, { layers: clusterLayers })[0]
        const clusterId = feature?.properties?.cluster_id
        const source = map.getSource(STATIONS_SOURCE) as GeoJSONSource | undefined
        if (feature && typeof clusterId === 'number' && source) {
          void source.getClusterExpansionZoom(clusterId).then((zoom) => {
            if (mapRef.current !== map) return
            const geometry = feature.geometry
            if (geometry.type !== 'Point') return
            map.easeTo({ center: geometry.coordinates as [number, number], zoom, duration: motionDuration(500) })
          })
          return
        }
      }
      const pointLayers = [STATION_POINTS_LAYER].filter((id) => map.getLayer(id))
      const city = map.queryRenderedFeatures(event.point, { layers: pointLayers })[0]?.properties?.city
      if (typeof city === 'string') onSelectRef.current(city)
    }

    const handleMove = (event: MapMouseEvent) => {
      const layers = [CLUSTERS_LAYER, STATION_POINTS_LAYER].filter((id) => map.getLayer(id))
      const interactive = layers.length
        ? map.queryRenderedFeatures(event.point, { layers }).length > 0
        : false
      map.getCanvas().style.cursor = interactive ? 'pointer' : ''
    }

    map.on('click', handleClick)
    map.on('mousemove', handleMove)

    // The container can settle after mount (lazy load, fonts, layout shifts),
    // which leaves the canvas short of its box. Track it explicitly.
    const observer = new ResizeObserver(() => map.resize())
    observer.observe(containerRef.current)
    map.once('load', () => map.resize())

    return () => {
      observer.disconnect()
      map.off('click', handleClick)
      map.off('mousemove', handleMove)
      map.remove()
      mapRef.current = null
    }
    // The map is created once; style swaps are handled below via setStyle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* setStyle wipes every custom source/layer, and calling it while the initial
     style is still loading crashes MapLibre's style-diff. Track what was last
     applied instead of "first run" — StrictMode re-runs effects on the same
     instance, so a first-run flag would be consumed before the remount. */
  const appliedStyleRef = useRef<BasemapId>(basemap)
  useEffect(() => {
    if (appliedStyleRef.current === basemap) return
    appliedStyleRef.current = basemap
    mapRef.current?.setStyle(basemapStyle(basemap))
  }, [basemap])

  /* Re-apply every custom source/layer after a style load or theme/basemap
     swap. Async because the SDF glyphs must exist before the symbol layers. */
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    let cancelled = false

    const apply = async () => {
      /* StrictMode double-mounts; a stale instance's listener must not touch the live map. */
      if (cancelled || mapRef.current !== map) return
      const panel = palette.panel
      const lineStrong = palette.lineStrong
      const line = palette.line
      const fg = palette.text
      const accent = palette.accent

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
        if (!map.getSource(TERRAIN_DEM_SOURCE)) {
          map.addSource(TERRAIN_DEM_SOURCE, {
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
              'hillshade-shadow-color': isDark
                ? MAP_PAINT.hillshadeShadow.dark
                : MAP_PAINT.hillshadeShadow.light,
            },
          })
        }
        /* Hillshade and the 3D mesh cannot run at once: two raster-dem consumers
           black out the Carto vector basemap past the DEM maxzoom (verified by
           bisect). 3D wins while it is on. */
        map.setLayoutProperty(
          HILLSHADE_LAYER,
          'visibility',
          mapLayers.hillshade && !mapLayers.terrain3d ? 'visible' : 'none',
        )

        if (mapLayers.terrain3d) {
          map.setTerrain({ source: TERRAIN_DEM_SOURCE, exaggeration: 1.2 })
          if (map.getPitch() < 1) map.easeTo({ pitch: 55, duration: motionDuration(600) })
        } else {
          map.setTerrain(null)
          if (map.getPitch() > 1) map.easeTo({ pitch: 0, duration: motionDuration(600) })
        }

        /* Building extrusions ride the vector tiles the basemap already ships
         * (render_height / render_min_height). The raster relief style has no
         * `carto` source, so the toggle is a no-op there. */
        if (!map.getLayer(BUILDINGS_LAYER) && map.getSource('carto')) {
          map.addLayer({
            id: BUILDINGS_LAYER,
            type: 'fill-extrusion',
            source: 'carto',
            'source-layer': 'building',
            minzoom: 14,
            paint: {
              'fill-extrusion-color': isDark ? MAP_PAINT.building.dark : MAP_PAINT.building.light,
              'fill-extrusion-height': ['coalesce', ['get', 'render_height'], 6],
              'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0],
              'fill-extrusion-opacity': 0.92,
              'fill-extrusion-vertical-gradient': true,
            },
          })
        }
        if (map.getLayer(BUILDINGS_LAYER)) {
          map.setLayoutProperty(
            BUILDINGS_LAYER,
            'visibility',
            mapLayers.buildings3d ? 'visible' : 'none',
          )
        }
        for (const flat of FLAT_BUILDING_LAYERS) {
          if (!map.getLayer(flat)) continue
          map.setLayoutProperty(flat, 'visibility', mapLayers.buildings3d ? 'none' : 'visible')
        }

        await ensureMapIcons(map)
        if (cancelled || mapRef.current !== map) return

        if (!map.getSource(STATIONS_SOURCE)) {
          map.addSource(STATIONS_SOURCE, stationSourceSpec(geoJsonRef.current))
        }

        if (!map.getLayer(CLUSTERS_LAYER)) {
          map.addLayer({
            id: CLUSTERS_LAYER,
            type: 'circle',
            source: STATIONS_SOURCE,
            filter: ['has', 'point_count'],
            paint: {
              'circle-color': panel,
              'circle-stroke-color': lineStrong,
              'circle-stroke-width': 1,
              'circle-radius': ['step', ['get', 'point_count'], 13, 5, 17, 15, 21],
            },
          })
        }
        if (!map.getLayer(CLUSTER_COUNT_LAYER)) {
          map.addLayer({
            id: CLUSTER_COUNT_LAYER,
            type: 'symbol',
            source: STATIONS_SOURCE,
            filter: ['has', 'point_count'],
            layout: {
              'text-field': ['get', 'point_count_abbreviated'],
              'text-font': ['Open Sans Regular'],
              'text-size': 11,
            },
            paint: { 'text-color': fg },
          })
        }

        if (!map.getLayer(STATION_HALO_LAYER)) {
          map.addLayer({
            id: STATION_HALO_LAYER,
            type: 'circle',
            source: STATIONS_SOURCE,
            filter: UNCLUSTERED_FILTER,
            paint: {
              'circle-color': panel,
              'circle-stroke-color': line,
              'circle-stroke-width': 1,
              'circle-radius': 9,
              'circle-opacity': 0.92,
            },
          })
        }
        if (!map.getLayer(STATION_SELECTED_LAYER)) {
          map.addLayer({
            id: STATION_SELECTED_LAYER,
            type: 'circle',
            source: STATIONS_SOURCE,
            filter: ['==', ['get', 'city'], selectedCityRef.current ?? ''],
            paint: {
              'circle-color': 'transparent',
              'circle-stroke-color': accent,
              'circle-stroke-width': 2,
              'circle-radius': 12,
            },
          })
        }
        if (!map.getLayer(STATION_WIND_LAYER)) {
          map.addLayer({
            id: STATION_WIND_LAYER,
            type: 'symbol',
            source: STATIONS_SOURCE,
            filter: [
              'all',
              ['!', ['has', 'point_count']],
              // Present-but-null wind must not draw a meaningless arrow.
              ['!=', ['get', 'wind_speed'], null],
              ['!=', ['get', 'wind_direction'], null],
            ],
            layout: {
              'icon-image': WIND_ARROW_IMAGE,
              // Meteorological direction is where the wind comes from.
              'icon-rotate': ['+', ['get', 'wind_direction'], 180],
              'icon-rotation-alignment': 'map',
              'icon-size': [
                'interpolate',
                ['linear'],
                ['get', 'wind_speed'],
                0, 0.45,
                15, 0.6,
                40, 0.8,
              ],
              'icon-allow-overlap': true,
              'icon-ignore-placement': true,
            },
            paint: {
              'icon-color': rampStepExpression('wind', 'wind_speed'),
              'icon-halo-color': fg,
              'icon-halo-width': 0.4,
              'icon-opacity': [
                'interpolate',
                ['linear'],
                ['get', 'wind_speed'],
                0, 0.35,
                30, 0.85,
              ],
            },
          })
        }
        if (!map.getLayer(STATION_POINTS_LAYER)) {
          map.addLayer({
            id: STATION_POINTS_LAYER,
            type: 'symbol',
            source: STATIONS_SOURCE,
            filter: UNCLUSTERED_FILTER,
            layout: {
              'icon-image': ['concat', WEATHER_IMAGE_PREFIX, ['get', 'weather_icon']],
              'icon-size': 0.42,
              'icon-allow-overlap': true,
              'icon-ignore-placement': true,
            },
            paint: {
              'icon-color': rampStepExpression('temp', 'temperature'),
              'icon-halo-color': fg,
              'icon-halo-width': 0.7,
            },
          })
        }

        if (mapLayers.radar && radarUrlRef.current) applyRadar(map, radarUrlRef.current)
        else if (map.getLayer(RADAR_LAYER)) {
          map.setLayoutProperty(RADAR_LAYER, 'visibility', 'none')
        }
      } catch {
        /* A failed tile source must not take the basemap down. */
      }
    }

    /* style.load (not styledata) is what the minimal repro proved reliable here. */
    if (map.isStyleLoaded()) void apply()
    else map.once('style.load', apply)
    return () => {
      cancelled = true
      map.off('style.load', apply)
    }
  }, [mapLayers, basemap, isDark, palette])

  /* Station data arrives separately from the style, so push it when it changes. */
  useEffect(() => {
    const map = mapRef.current
    const source = map?.getSource(STATIONS_SOURCE) as GeoJSONSource | undefined
    if (source) source.setData(geoJson)
  }, [geoJson])

  /* Selected-city emphasis layer (re-created on style swaps). */
  useEffect(() => {
    const map = mapRef.current
    if (map?.getLayer(STATION_SELECTED_LAYER)) {
      map.setFilter(STATION_SELECTED_LAYER, ['==', ['get', 'city'], selectedCity ?? ''])
    }
  }, [selectedCity, mapLayers, basemap])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !selectedCity) return
    const feature = geoJson.features.find((candidate) => candidate.properties.city === selectedCity)
    if (!feature) return
    map.easeTo({
      center: feature.geometry.coordinates as [number, number],
      zoom: Math.max(map.getZoom(), 7),
      duration: motionDuration(700),
    })
  }, [selectedCity, geoJson])

  /* Radar frame changes swap the raster tiles in place. */
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (!mapLayers.radar) {
      if (map.getLayer(RADAR_LAYER)) map.setLayoutProperty(RADAR_LAYER, 'visibility', 'none')
      return
    }
    if (radarUrlRef.current) applyRadar(map, radarUrlRef.current)
  }, [mapLayers.radar, radar.current, radar.host, basemap])

  const layerOptions = [
    { key: 'hillshade' as const, label: 'Relieve' },
    { key: 'radar' as const, label: 'Radar de lluvia' },
    { key: 'terrain3d' as const, label: 'Terreno 3D' },
    { key: 'buildings3d' as const, label: 'Edificios 3D' },
  ]

  return (
    <div className={cn('relative h-full w-full', className)}>
      <div
        ref={containerRef}
        className="h-full w-full bg-panel-2"
        aria-label="Mapa de estaciones meteorológicas"
      />

      <div className="absolute right-2 top-2 z-10 w-44 rounded-[3px] border border-line bg-panel p-2.5">
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
                onChange={(event) => {
                  /* Relief and 3D terrain are the same DEM presented two ways:
                     keep exactly one active (radio-like). Radar and buildings
                     are independent. */
                  if (option.key === 'terrain3d' && event.target.checked) {
                    setMapLayers({ ...mapLayers, terrain3d: true, hillshade: false })
                  } else if (option.key === 'hillshade' && event.target.checked) {
                    setMapLayers({ ...mapLayers, hillshade: true, terrain3d: false })
                  } else {
                    setMapLayers({ ...mapLayers, [option.key]: event.target.checked })
                  }
                }}
              />
              {option.label}
            </label>
          ))}
        </div>

        <label className="label mt-2.5 block" htmlFor="basemap-select">
          Base
        </label>
        <select
          id="basemap-select"
          value={basemap}
          onChange={(event) => setBasemap(event.target.value as BasemapId)}
          className="mt-1 h-7 w-full rounded-[3px] border border-line bg-panel-2 px-1.5 text-[16px] leading-none text-fg-2 focus:border-accent focus:outline-none sm:text-xs"
        >
          <option value="positron">Claro (Carto)</option>
          <option value="dark">Oscuro (Carto)</option>
          <option value="topo">Relieve (OpenTopoMap)</option>
        </select>

        <div className="mt-2.5 border-t border-line pt-2">
          <div className="label mb-1">Temperatura</div>
          <div className="flex h-2 overflow-hidden rounded-[2px]" aria-hidden="true">
            {Array.from({ length: 7 }, (_, step) => (
              <span key={step} className="flex-1" style={{ background: rampToken('temp', step) }} />
            ))}
          </div>
          <div className="nums mt-0.5 flex justify-between text-[9px] text-fg-3">
            <span>{RAMP_DOMAINS.temp[0]}°</span>
            <span>icono = temp.</span>
            <span>{RAMP_DOMAINS.temp[1]}°</span>
          </div>
          <div className="label mb-1 mt-2">Viento (flecha)</div>
          <div className="flex h-2 overflow-hidden rounded-[2px]" aria-hidden="true">
            {Array.from({ length: 4 }, (_, step) => (
              <span key={step} className="flex-1" style={{ background: rampToken('wind', step) }} />
            ))}
          </div>
          <div className="nums mt-0.5 flex justify-between text-[9px] text-fg-3">
            <span>0</span>
            <span>tamaño = vel.</span>
            <span>{RAMP_DOMAINS.wind[1]} km/h</span>
          </div>
        </div>

        {mapLayers.radar && radar.status === 'unavailable' ? (
          <p className="mt-1.5 text-[10px] leading-tight text-warn">Radar no disponible ahora.</p>
        ) : null}
      </div>

      {mapLayers.radar ? (
        <RadarControls
          frames={radar.frames}
          index={radar.index}
          onIndex={radar.setIndex}
          playing={radar.playing}
          onTogglePlay={radar.togglePlay}
          reducedMotion={radar.reducedMotion}
          className="absolute bottom-2 left-2 z-10 max-w-[calc(100%-1rem)]"
        />
      ) : null}
    </div>
  )
}

/* Latest public RainViewer frame; added on top of the basemap and swapped in
 * place with `setTiles` so playback never flickers. */
function applyRadar(map: maplibregl.Map, tileUrl: string): void {
  try {
    if (!map.getSource(RADAR_SOURCE)) {
      map.addSource(RADAR_SOURCE, {
        type: 'raster',
        tiles: [tileUrl],
        tileSize: 256,
        maxzoom: 10,
        attribution: 'Radar: RainViewer',
      })
    } else {
      ;(map.getSource(RADAR_SOURCE) as RasterTileSource).setTiles([tileUrl])
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
  } catch {
    /* Radar is best-effort; the basemap and stations stay usable. */
  }
}
