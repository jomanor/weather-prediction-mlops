import { NavLink, Outlet } from 'react-router-dom'

import { NAV_ITEMS } from '@/app/registry'
import { StationSearch } from '@/components/shell/StationSearch'
import { ThemeToggle } from '@/components/shell/ThemeToggle'
import { UnitToggle } from '@/components/shell/UnitToggle'
import { cn } from '@/lib/cn'

export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col bg-bg">
      <header className="sticky top-0 z-40 border-b border-line bg-panel">
        <div className="flex h-12 items-center gap-6 px-4">
          <NavLink to="/" className="shrink-0 text-[15px] font-semibold leading-none text-fg">
            meteoml
          </NavLink>

          <nav aria-label="Secciones" className="hidden h-full items-center sm:flex">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex h-full items-center border-b-2 px-3 text-[13px] transition-colors',
                    isActive
                      ? 'border-fg font-medium text-fg'
                      : 'border-transparent text-fg-2 hover:text-fg',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <div className="hidden md:block">
              <StationSearch />
            </div>
            <div className="hidden sm:block">
              <UnitToggle />
            </div>
            <ThemeToggle />
          </div>
        </div>
        <div className="border-t border-line px-4 py-2 md:hidden">
          <StationSearch />
        </div>
      </header>

      <main className="min-w-0 flex-1 pb-16 md:pb-0">
        <Outlet />
      </main>

      <footer className="border-t border-line px-4 py-2.5 text-[11px] text-fg-3">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-1">
          <span>Datos: Open-Meteo · AEMET OpenData · Radar: RainViewer · Relieve: AWS Terrain</span>
          <span className="nums">meteoml · {new Date().getFullYear()}</span>
        </div>
      </footer>

      <nav
        aria-label="Secciones"
        className="fixed inset-x-0 bottom-0 z-40 grid border-t border-line bg-panel md:hidden"
        style={{ gridTemplateColumns: `repeat(${NAV_ITEMS.length}, minmax(0, 1fr))` }}
      >
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                cn(
                  'flex flex-col items-center gap-1 py-2 text-[10px]',
                  isActive ? 'text-fg' : 'text-fg-3',
                )
              }
            >
              {({ isActive }) => (
                <>
                  {Icon ? (
                    <Icon
                      aria-hidden
                      className={cn('h-3.5 w-3.5', isActive ? 'text-fg' : 'text-fg-3')}
                    />
                  ) : (
                    <span
                      aria-hidden
                      className={cn('h-0.5 w-4 rounded-full', isActive ? 'bg-fg' : 'bg-transparent')}
                    />
                  )}
                  {item.label}
                </>
              )}
            </NavLink>
          )
        })}
      </nav>
    </div>
  )
}
