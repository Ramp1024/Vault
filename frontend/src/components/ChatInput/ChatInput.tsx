import { useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'

type ChatInputProps = {
  onSend: (message: string) => Promise<void>
  disabled: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('')

  const submit = async (event?: FormEvent) => {
    event?.preventDefault()

    const trimmed = message.trim()
    if (!trimmed || disabled) {
      return
    }

    setMessage('')
    await onSend(trimmed)
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void submit()
    }
  }

  return (
    <form className="composer" onSubmit={(event) => void submit(event)}>
      <textarea
        className="composer-input"
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Write a message…"
        disabled={disabled}
        rows={1}
      />
      <div className="composer-footer">
        <span className="composer-hint">
          <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for newline
        </span>
        <button
          type="submit"
          className="composer-send"
          disabled={disabled || !message.trim()}
          aria-label="Send message"
        >
          {disabled ? (
            <span className="composer-spinner" aria-hidden="true" />
          ) : (
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <path
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 19V5M6 11l6-6 6 6"
              />
            </svg>
          )}
        </button>
      </div>
    </form>
  )
}
