import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../../pages/ChatPage/ChatPage'
import { MessageBubble } from '../MessageBubble/MessageBubble'

// A prefill drops a ready-to-edit sentence into the composer with the pluggable
// part pre-selected, so the user can immediately type over it (or send as-is).
export type PromptPrefill = {
  text: string
  selectionStart: number
  selectionEnd: number
}

type ChatWindowProps = {
  messages: ChatMessage[]
  isStreaming: boolean
  onPromptSelect?: (prefill: PromptPrefill) => void
  onRegenerate?: (assistantId: string) => void
}

// Generic, reusable templates. Each has a fixed frame (`before`/`after`) and a
// pluggable example value the user swaps out. The example doubles as a valid
// query, so pressing Enter without editing still asks a real question.
type PromptTemplate = {
  before: string
  placeholder: string
  after: string
}

const EXAMPLE_PROMPTS: PromptTemplate[] = [
  { before: 'What did I do on ', placeholder: 'August 15', after: '?' },
  { before: 'Show me my ', placeholder: 'System Design', after: ' notes' },
  { before: 'Explain ', placeholder: 'tree shaking', after: ' from my notes' },
  { before: 'Why did I choose ', placeholder: 'FastAPI', after: '?' },
]

const toPrefill = (template: PromptTemplate): PromptPrefill => ({
  text: `${template.before}${template.placeholder}${template.after}`,
  selectionStart: template.before.length,
  selectionEnd: template.before.length + template.placeholder.length,
})

export function ChatWindow({ messages, isStreaming, onPromptSelect, onRegenerate }: ChatWindowProps) {
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
              <li key={`${prompt.before}${prompt.placeholder}`}>
                <button
                  type="button"
                  className="prompt-row"
                  onClick={() => onPromptSelect?.(toPrefill(prompt))}
                >
                  <span className="prompt-row-text">
                    {prompt.before}
                    <span className="prompt-placeholder">{prompt.placeholder}</span>
                    {prompt.after}
                  </span>
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
        <MessageBubble key={message.id} message={message} onRegenerate={onRegenerate} />
      ))}
      <div ref={endRef} aria-hidden="true" />
    </section>
  )
}
