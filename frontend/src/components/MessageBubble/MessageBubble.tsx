import type { ChatCitation, ChatRole, ChatSource } from '../../pages/ChatPage/ChatPage'

type MessageBubbleProps = {
  role: ChatRole
  content: string
  sources?: ChatSource[]
  citations?: ChatCitation[]
}

export function MessageBubble({ role, content, sources, citations }: MessageBubbleProps) {
  const sideClass = role === 'user' ? 'user' : 'assistant'
  const hasSources = role === 'assistant' && sources && sources.length > 0
  const hasCitations = role === 'assistant' && citations && citations.length > 0

  return (
    <article className={`message-row ${sideClass}`}>
      <div className="message-bubble">
        <p>{content || '...'}</p>
        {hasCitations ? (
          <div className="message-sources">
            <span className="message-sources-label">Citations</span>
            <ul className="source-chips">
              {citations!.map((citation) => (
                <li key={citation.chunk_id}>
                  <span className="source-chip">
                    {`[${citation.reference_id}] ${citation.title}`}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
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
                    {source.reference_id != null
                      ? `[${source.reference_id}] ${source.title}`
                      : source.title}
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
