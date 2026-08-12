import { useState } from 'react'
import type { ChatMessage } from '../../pages/ChatPage/ChatPage'
import { LoadingState } from '../LoadingState/LoadingState'
import { ThinkingTrace } from '../ThinkingTrace/ThinkingTrace'

type MessageBubbleProps = {
  message: ChatMessage
}

const CopyIcon = () => (
  <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
    <g fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="11" height="11" rx="2.5" />
      <path d="M5 15V6a2 2 0 0 1 2-2h8" />
    </g>
  </svg>
)

const CheckIcon = () => (
  <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
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

export function MessageBubble({ message }: MessageBubbleProps) {
  const { role, content, sources, citations, trace, phase, startedAt, finishedAt } = message
  const [copied, setCopied] = useState(false)

  if (role === 'user') {
    return (
      <article className="msg msg-user">
        <div className="msg-user-pill">
          <p>{content}</p>
        </div>
      </article>
    )
  }

  const isLoading = phase === 'loading' && !content
  const isStreaming = phase === 'streaming'
  const isDone = phase === 'done' || phase == null
  const hasSources = sources && sources.length > 0
  const hasCitations = citations && citations.length > 0

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      // Clipboard may be unavailable; ignore.
    }
  }

  return (
    <article className="msg msg-assistant">
      <div className="msg-role">
        <span className="msg-role-dot" aria-hidden="true" />
        Vault
      </div>

      {trace && trace.length > 0 ? (
        <ThinkingTrace steps={trace} startedAt={startedAt ?? Date.now()} finishedAt={finishedAt} />
      ) : null}

      {isLoading ? (
        <LoadingState label="Retrieving" startedAt={startedAt ?? Date.now()} />
      ) : (
        <div className="msg-body">
          <p>
            {content}
            {isStreaming ? <span className="stream-caret" aria-hidden="true" /> : null}
          </p>
        </div>
      )}

      {hasCitations ? (
        <div className="msg-refs">
          <span className="msg-refs-label">Citations</span>
          <ul className="ref-chips">
            {citations!.map((citation) => (
              <li key={citation.chunk_id}>
                <span className="ref-chip">
                  <span className="ref-chip-index">{citation.reference_id}</span>
                  {citation.title}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {hasSources ? (
        <div className="msg-refs">
          <span className="msg-refs-label">{sources!.length} sources</span>
          <ul className="ref-chips">
            {sources!.map((source) => (
              <li key={source.chunk_id}>
                <span
                  className="ref-chip"
                  title={`${source.snippet}\n\n(score ${source.score.toFixed(4)})`}
                >
                  {source.reference_id != null ? (
                    <span className="ref-chip-index">{source.reference_id}</span>
                  ) : null}
                  {source.title}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {isDone && content ? (
        <div className="msg-actions">
          <button type="button" className="msg-action" onClick={handleCopy}>
            {copied ? <CheckIcon /> : <CopyIcon />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      ) : null}
    </article>
  )
}
