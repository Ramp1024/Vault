import { createPortal } from 'react-dom'

export type ToastVariant = 'success' | 'error'

type ToastProps = {
    variant: ToastVariant
    message: string
    onDismiss: () => void
}

const CloseIcon = () => (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
        <path
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            d="M6 6l12 12M18 6L6 18"
        />
    </svg>
)

const SuccessIcon = () => (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
        <path
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M5 12.5l4.5 4.5L19 7"
        />
    </svg>
)

const ErrorIcon = () => (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
        <g fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="9" />
            <path d="M12 7.5v5M12 16h.01" />
        </g>
    </svg>
)

/**
 * A single dismissible toast anchored to the viewport's top-right. It never
 * auto-hides — it stays until the user clicks the close button.
 */
export function Toast({ variant, message, onDismiss }: ToastProps) {
    return createPortal(
        <div className={`toast toast-${variant}`} role="status" aria-live="polite">
            <span className="toast-icon" aria-hidden="true">
                {variant === 'success' ? <SuccessIcon /> : <ErrorIcon />}
            </span>
            <span className="toast-message">{message}</span>
            <button
                type="button"
                className="toast-close"
                onClick={onDismiss}
                aria-label="Dismiss notification"
            >
                <CloseIcon />
            </button>
        </div>,
        document.body,
    )
}
