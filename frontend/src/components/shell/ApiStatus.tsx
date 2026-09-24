import { useHealth } from '@/api/queries'
import { StatusDot } from '@/components/ui/Badge'

/** Live indicator for the backend, so an outage is visible rather than silent. */
export function ApiStatus() {
  const { data, isError, isLoading } = useHealth()

  if (isLoading) {
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-fg-3">
        <StatusDot tone="neutral" pulse />
        Conectando
      </span>
    )
  }

  if (isError || !data) {
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-bad" title="La API no responde">
        <StatusDot tone="bad" />
        API no disponible
      </span>
    )
  }

  return (
    <span
      className="flex items-center gap-1.5 text-[11px] text-fg-3"
      title={`${data.service} v${data.version}`}
    >
      <StatusDot tone="ok" pulse />
      <span className="nums">API v{data.version}</span>
    </span>
  )
}
