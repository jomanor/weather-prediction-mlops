import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'

import { App } from '@/App'
import { PreferencesProvider } from '@/app/preferences'

import '@/styles/index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      /* Finding 15: the poll is the only refresh path, so the moment the viewer
         comes back to the tab is when re-reading is worth paying for. React Query
         only refetches queries that are already stale (60 s here), so a quick
         tab flip inside the window costs nothing. */
      refetchOnWindowFocus: true,
      staleTime: 60_000,
      retry: 1,
    },
  },
})

const container = document.getElementById('root')
if (!container) throw new Error('Root container #root not found')

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <PreferencesProvider>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </PreferencesProvider>
    </QueryClientProvider>
  </StrictMode>,
)
