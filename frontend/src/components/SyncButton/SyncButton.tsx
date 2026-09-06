import { useCallback, useState } from 'react'
import { Toast } from '../Toast/Toast'
import type { ToastVariant } from '../Toast/Toast'

type SyncStatus = 'idle' | 'syncing' | 'success' | 'error'

type ToastState = {
    variant: ToastVariant
    message: string
}

type SyncResult = {
    documents_processed: number
    documents_skipped: number
    chunks_created: number
    embeddings_generated: number
    vectors_upserted: number
    duration: number
}

const SyncIcon = ({ spinning }: { spinning: boolean }) => (
    <svg
        viewBox="0 0 24 24"
        width="15"
        height="15"
        aria-hidden="true"
        className={spinning ? 'sync-icon spinning' : 'sync-icon'}
    >
        <g fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 11a8 8 0 0 0-14.3-4.5M4 4v3.5h3.5" />
            <path d="M4 13a8 8 0 0 0 14.3 4.5M20 20v-3.5h-3.5" />
        </g>
    </svg>
)

export function SyncButton() {
    const [status, setStatus] = useState<SyncStatus>('idle')
    const [toast, setToast] = useState<ToastState | null>(null)

    const handleSync = useCallback(async () => {
        if (status === 'syncing') {
            return
        }
        setStatus('syncing')

        try {
            const response = await fetch('/api/sync', { method: 'POST' })
            if (!response.ok) {
                let message = `Sync failed (status ${response.status})`
                try {
                    const body = (await response.json()) as { detail?: string }
                    if (body.detail) {
                        message = body.detail
                    }
                } catch {
                    // Ignore unparseable error bodies and keep the status message.
                }
                throw new Error(message)
            }

            const result = (await response.json()) as SyncResult
            setStatus('success')
            setToast({
                variant: 'success',
                message: `Sync complete — processed ${result.documents_processed}, skipped ${result.documents_skipped} in ${result.duration}s`,
            })
        } catch (error) {
            setStatus('error')
            setToast({
                variant: 'error',
                message: error instanceof Error ? error.message : 'Sync failed.',
            })
        }
    }, [status])

    const label = status === 'syncing' ? 'Syncing…' : 'Sync'

    return (
        <>
            <button
                type="button"
                className="sync-button"
                onClick={handleSync}
                disabled={status === 'syncing'}
                aria-busy={status === 'syncing'}
            >
                <SyncIcon spinning={status === 'syncing'} />
                <span>{label}</span>
            </button>
            {toast && (
                <Toast
                    variant={toast.variant}
                    message={toast.message}
                    onDismiss={() => setToast(null)}
                />
            )}
        </>
    )
}
