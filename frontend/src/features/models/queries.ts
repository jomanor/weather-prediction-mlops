import { queryKeys, useApiQuery } from '@/api/queries'
import { modelsResponseSchema } from '@/api/schemas'

/** Registry of trained artefacts served by `/api/models`. */
export const useModels = () =>
  useApiQuery(queryKeys.models, '/models', modelsResponseSchema, { staleTime: 10 * 60_000 })
