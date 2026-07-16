import type { ChatRole } from '../../pages/ChatPage/ChatPage'

type MessageBubbleProps = {
  role: ChatRole
  content: string
}

export function MessageBubble({ role, content }: MessageBubbleProps) {
  const sideClass = role === 'user' ? 'user' : 'assistant'

  return (
    <article className={`message-row ${sideClass}`}>
      <div className="message-bubble">
        <p>{content || '...'}</p>
      </div>
    </article>
  )
}
