import { useElapsed } from '../../hooks/useElapsed'
import { formatDuration } from '../../utils/time'

type LoadingStateProps = {
    label?: string
    startedAt: number
}

const CELLS = Array.from({ length: 9 })

/**
 * Pixel-grid loader with a shimmering label and live elapsed time — shown while
 * the assistant is retrieving before any answer text has streamed in.
 */
export function LoadingState({ label = 'Churning', startedAt }: LoadingStateProps) {
    const elapsed = useElapsed(startedAt, true)

    return (
        <div className="loading-state" aria-live="polite">
            <span className="pixel-grid" aria-hidden="true">
                {CELLS.map((_, index) => (
                    <span
                        key={index}
                        className="pixel-cell"
                        style={{
                            animationDelay: `${((index % 3) + Math.floor(index / 3)) * 0.11}s`,
                        }}
                    />
                ))}
            </span>
            <span className="loading-label">{label}</span>
            <span className="loading-elapsed">{formatDuration(elapsed)}</span>
        </div>
    )
}
