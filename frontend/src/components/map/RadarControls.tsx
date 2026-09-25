import { Pause, Play } from 'lucide-react'

import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/cn'
import { formatTime } from '@/lib/format'

import type { RadarFrame } from './radar'

interface RadarControlsProps {
  frames: RadarFrame[]
  index: number
  onIndex: (index: number) => void
  playing: boolean
  onTogglePlay: () => void
  /** Reduced motion blocks playback; the scrubber stays usable. */
  reducedMotion?: boolean
  className?: string
}

/**
 * Time scrubber for the radar loop. Manual by default — nothing autoplays, and
 * under `prefers-reduced-motion` the loop is disabled outright (the scrubber
 * still steps through frames one at a time).
 */
export function RadarControls({
  frames,
  index,
  onIndex,
  playing,
  onTogglePlay,
  reducedMotion = false,
  className,
}: RadarControlsProps) {
  if (frames.length === 0) return null
  const current = frames[index] ?? frames[frames.length - 1]
  const last = frames.length - 1
  const stamp = new Date(current.time * 1000).toISOString()
  const playbackDisabled = reducedMotion || frames.length < 2

  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-[3px] border border-line bg-panel px-2 py-1.5',
        className,
      )}
      role="group"
      aria-label="Reproducción del radar"
    >
      <button
        type="button"
        onClick={onTogglePlay}
        disabled={playbackDisabled}
        aria-label={
          reducedMotion
            ? 'Reproducción desactivada por reducción de movimiento'
            : playing
              ? 'Pausar radar'
              : 'Reproducir radar'
        }
        aria-pressed={playing}
        title={reducedMotion ? 'Movimiento reducido activo' : undefined}
        className="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-line text-fg-2 hover:border-line-strong hover:text-fg disabled:cursor-not-allowed disabled:opacity-40"
      >
        {playing ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
      </button>

      <input
        type="range"
        min={0}
        max={last}
        step={1}
        value={index}
        onChange={(event) => onIndex(Number(event.target.value))}
        aria-label="Fotograma del radar"
        aria-valuetext={formatTime(stamp)}
        className="h-1 w-28 cursor-pointer accent-accent"
      />

      <time className="nums w-[68px] shrink-0 text-[10px] text-fg-2" dateTime={stamp}>
        {formatTime(stamp)}
      </time>
      <span className="nums text-[10px] text-fg-3">
        {index + 1}/{frames.length}
      </span>
      {current.kind === 'nowcast' ? <Badge tone="accent">Previsión</Badge> : null}
    </div>
  )
}
