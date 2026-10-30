import { queryKeys, useApiQuery } from '@/api/queries'
import { stationCollectionSchema, stationsResponseSchema } from '@/api/schemas'

/**
 * Resumen queries: the national station list and the bulk map layer.
 *
 * Reading only. The registry writes (`POST /cities`, `DELETE /cities/{name}`)
 * are gated on the backend, so the UI does not offer them; `/geo/search` lost
 * its only consumer with the city manager.
 */

export const useStations = () =>
  useApiQuery(queryKeys.stations, '/weather/current', stationsResponseSchema, {
    refetchInterval: 5 * 60_000,
  })

/** Bulk GeoJSON for the clustered map layer (Batch 2 endpoint). */
export const useMapStations = () =>
  useApiQuery(queryKeys.mapStations, '/map/stations', stationCollectionSchema, {
    refetchInterval: 5 * 60_000,
  })
