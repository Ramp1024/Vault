import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../../pages/ChatPage/ChatPage'
import { MessageBubble } from '../MessageBubble/MessageBubble'
import { TypingIndicator } from '../TypingIndicator/TypingIndicator'

type ChatWindowProps = {
  messages: ChatMessage[]
  isStreaming: boolean
}

export function ChatWindow({ messages, isStreaming }: ChatWindowProps) {
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isStreaming])

  if (messages.length === 0) {
    return (
      <section className="chat-window empty">
        <p>Start by asking a question.</p>
      </section>
    )
  }

  const lastMessage = messages[messages.length - 1]
  const showTyping =
    isStreaming && lastMessage?.role === 'assistant' && !lastMessage.content

  return (
    <section className="chat-window">
      {messages.map((message) => (
        <MessageBubble key={message.id} role={message.role} content={message.content} />
      ))}
      {showTyping ? <TypingIndicator /> : null}
      <div ref={endRef} aria-hidden="true" />
    </section>
  )
}
