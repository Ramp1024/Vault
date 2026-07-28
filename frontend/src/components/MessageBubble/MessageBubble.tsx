import type { ChatRole, ChatSource } from '../../pages/ChatPage/ChatPage'

type MessageBubbleProps = {
  role: ChatRole
  content: string
  sources?: ChatSource[]
}

export function MessageBubble({ role, content, sources }: MessageBubbleProps) {
  const sideClass = role === 'user' ? 'user' : 'assistant'
  const hasSources = role === 'assistant' && sources && sources.length > 0

  return (
    <article className={`message-row ${sideClass}`}>
      <div className="message-bubble">
        <p>{content || '...'}</p>
        {hasSources ? (
          <div className="message-sources">
            <span className="message-sources-label">Sources</span>
            <ul className="source-chips">
              {sources!.map((source) => (
                <li key={source.chunk_id}>
                  <span
                    className="source-chip"
                    title={`${source.snippet}\n\n(score ${source.score.toFixed(4)})`}
                  >
                    {source.title}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </article>
  )
}
