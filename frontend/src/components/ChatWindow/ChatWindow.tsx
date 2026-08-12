import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../../pages/ChatPage/ChatPage'
import { MessageBubble } from '../MessageBubble/MessageBubble'

type ChatWindowProps = {
  messages: ChatMessage[]
  isStreaming: boolean
  onPromptSelect?: (prompt: string) => void
}

const EXAMPLE_PROMPTS = [
  'Summarize my most recent notes',
  'What decisions did I capture last week?',
  'Find open questions across my Vault',
  'Explain a concept from my knowledge base',
]

export function ChatWindow({ messages, isStreaming, onPromptSelect }: ChatWindowProps) {
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isStreaming])

  if (messages.length === 0) {
    return (
      <section className="chat-window empty">
        <div className="empty-state">
          <p className="empty-state-eyebrow">Get started</p>
          <h2>Ask your Vault anything</h2>
          <p className="empty-state-sub">
            Answers are grounded in your indexed knowledge base.
          </p>
          <ul className="empty-state-prompts">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <li key={prompt}>
                <button
                  type="button"
                  className="prompt-row"
                  onClick={() => onPromptSelect?.(prompt)}
                >
                  <span className="prompt-row-text">{prompt}</span>
                  <span className="prompt-row-arrow" aria-hidden="true">
                    <svg viewBox="0 0 24 24" width="14" height="14">
                      <path
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M5 12h14M13 6l6 6-6 6"
                      />
                    </svg>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </section>
    )
  }

  return (
    <section className="chat-window">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      <div ref={endRef} aria-hidden="true" />
    </section>
  )
}
