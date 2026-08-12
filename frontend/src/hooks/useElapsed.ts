import { useEffect, useState } from 'react'

/**
 * Returns elapsed milliseconds since `startedAt`. While `running` is true the
 * value ticks live; once stopped it freezes at `frozenAt - startedAt`.
 */
export function useElapsed(
    startedAt: number | undefined,
    running: boolean,
    frozenAt?: number,
): number {
    const [now, setNow] = useState(() => Date.now())

    useEffect(() => {
        if (!running || startedAt == null) {
            return
        }
        setNow(Date.now())
        const id = window.setInterval(() => setNow(Date.now()), 100)
        return () => window.clearInterval(id)
    }, [running, startedAt])

    if (startedAt == null) {
        return 0
    }
    const end = running ? now : frozenAt ?? now
    return Math.max(0, end - startedAt)
}
