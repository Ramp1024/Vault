import { useEffect, useState } from 'react'
import type { TraceStep } from '../../pages/ChatPage/ChatPage'
import { useElapsed } from '../../hooks/useElapsed'
import { formatDuration } from '../../utils/time'

type ThinkingTraceProps = {
    steps: TraceStep[]
    startedAt: number
    finishedAt?: number
}

const SparkIcon = () => (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
        <path
            fill="currentColor"
            d="M12 2.5c.4 2.9 1.6 4.1 4.5 4.5-2.9.4-4.1 1.6-4.5 4.5-.4-2.9-1.6-4.1-4.5-4.5 2.9-.4 4.1-1.6 4.5-4.5ZM18.5 13c.25 1.7 1 2.45 2.7 2.7-1.7.25-2.45 1-2.7 2.7-.25-1.7-1-2.45-2.7-2.7 1.7-.25 2.45-1 2.7-2.7Z"
        />
    </svg>
)

const ChevronIcon = () => (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
        <path
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M6 9l6 6 6-6"
        />
    </svg>
)

/**
 * Collapsible reasoning trace. While running it stays open and shows live steps;
 * once finished it collapses to a "Thought for Xs" summary the user can reopen.
 */
export function ThinkingTrace({ steps, startedAt, finishedAt }: ThinkingTraceProps) {
    const running = finishedAt == null
    const elapsed = useElapsed(startedAt, running, finishedAt)
    const [open, setOpen] = useState(running)

    useEffect(() => {
        if (finishedAt != null) {
            setOpen(false)
        }
    }, [finishedAt])

    return (
        <div className={`thinking ${open ? 'open' : ''}`}>
            <button
                type="button"
                className="thinking-header"
                onClick={() => setOpen((value) => !value)}
                aria-expanded={open}
            >
                <span className="thinking-spark">
                    <SparkIcon />
                </span>
                <span className="thinking-title">
                    {running ? (
                        <>
                            Thinking
                            <span className="thinking-dots" aria-hidden="true">
                                <span>.</span>
                                <span>.</span>
                                <span>.</span>
                            </span>
                        </>
                    ) : (
                        `Thought for ${formatDuration(elapsed)}`
                    )}
                </span>
                <span className="thinking-chevron">
                    <ChevronIcon />
                </span>
            </button>
            {open ? (
                <ol className="thinking-steps">
                    {steps.map((step) => (
                        <li key={step.id} className={`thinking-step ${step.status}`}>
                            <span className="thinking-step-marker" aria-hidden="true" />
                            <span className="thinking-step-label">{step.label}</span>
                        </li>
                    ))}
                </ol>
            ) : null}
        </div>
    )
}
