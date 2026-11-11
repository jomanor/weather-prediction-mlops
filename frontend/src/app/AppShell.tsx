import { NavLink, Outlet } from 'react-router-dom'

import { NAV_ITEMS } from '@/app/registry'
import { StationSearch } from '@/components/shell/StationSearch'
import { ThemeToggle } from '@/components/shell/ThemeToggle'
import { UnitToggle } from '@/components/shell/UnitToggle'
import { cn } from '@/lib/cn'

/**
 * The shell: brand, the three-section navigation, the one station search and the
 * preferences. There is a single search instance — inline in the bar from `md`
 * up, a full-bleed second row below it on a phone — so the results list has one
 * id and the `/` and ⌘K accelerators are bound once (`StationSearch`).
 */
export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col bg-bg">
      <header className="sticky top-0 z-40 border-b border-line bg-panel">
        <div className="flex flex-wrap items-center gap-x-6 px-4">
          <NavLink
            to="/"
            className="flex h-12 shrink-0 items-center text-[15px] font-semibold leading-none text-fg"
          >
            meteoml
          </NavLink>

          <nav aria-label="Secciones" className="hidden h-12 items-center sm:flex">
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

          <div className="ml-auto flex h-12 items-center gap-3 md:ml-0">
            <div className="hidden sm:block">
              <UnitToggle />
            </div>
            <ThemeToggle />
          </div>

          {/* `basis-full` wraps the search onto its own row under `md`; the
              negative margin makes that row's divider run edge to edge. From
              `md` up it is a normal inline item pushed to the right. It comes
              after the controls in the DOM so focus order matches what a phone
              shows (search on the second row, below the controls). */}
          <div className="-mx-4 w-full basis-full border-t border-line px-4 py-2 md:mx-0 md:ml-auto md:w-auto md:basis-auto md:border-0 md:px-0 md:py-0">
            <StationSearch />
          </div>
        </div>
      </header>

      {/* Same offset as the bottom nav (see `--nav-h`), so the last row of a page
          is never hidden behind the fixed nav on a phone. */}
      <main className="min-w-0 flex-1 pb-[calc(var(--nav-h)+var(--safe-bottom)+0.5rem)] md:pb-0">
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
        style={{
          gridTemplateColumns: `repeat(${NAV_ITEMS.length}, minmax(0, 1fr))`,
          /* `--nav-h` is the nav's own height and the home indicator is added
             on top. Three items share the row, so each is a full-width touch
             target well above 44 px at 360 px. */
          height: 'calc(var(--nav-h) + var(--safe-bottom))',
          paddingBottom: 'var(--safe-bottom)',
        }}
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
                  'flex flex-col items-center justify-center gap-1 text-[10px]',
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
