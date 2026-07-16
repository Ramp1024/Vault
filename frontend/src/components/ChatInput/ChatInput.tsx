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
    <form className="chat-input" onSubmit={(event) => void submit(event)}>
      <textarea
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Ask about your Vault content..."
        disabled={disabled}
        rows={2}
      />
      <button type="submit" disabled={disabled || !message.trim()}>
        {disabled ? 'Thinking...' : 'Send'}
      </button>
    </form>
  )
}
