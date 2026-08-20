import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import type { PromptPrefill } from '../ChatWindow/ChatWindow'

type ChatInputProps = {
  onSend: (message: string) => Promise<void>
  disabled: boolean
  // A bumped `nonce` re-applies the same prefill even if its text is unchanged.
  prefill?: (PromptPrefill & { nonce: number }) | null
}

export function ChatInput({ onSend, disabled, prefill }: ChatInputProps) {
  const [message, setMessage] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  // Selection to apply once the prefilled value is actually committed to the DOM.
  const pendingSelection = useRef<{ start: number; end: number } | null>(null)

  // Drop a template into the composer. We only queue the value + desired
  // selection here; the selection itself is applied in the layout effect below,
  // after React has committed the new value to the textarea.
  useEffect(() => {
    if (!prefill) {
      return
    }
    pendingSelection.current = {
      start: prefill.selectionStart,
      end: prefill.selectionEnd,
    }
    setMessage(prefill.text)
  }, [prefill])

  // Runs synchronously after the DOM value updates but before paint, so the
  // pluggable part is selected on the first click (no render-timing race).
  useLayoutEffect(() => {
    const selection = pendingSelection.current
    const textarea = textareaRef.current
    if (!selection || !textarea) {
      return
    }
    pendingSelection.current = null
    textarea.focus()
    textarea.setSelectionRange(selection.start, selection.end)
  }, [message])

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
        ref={textareaRef}
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
