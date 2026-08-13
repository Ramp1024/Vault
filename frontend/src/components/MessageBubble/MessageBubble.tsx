import { useState } from 'react'
import type { ChatMessage } from '../../pages/ChatPage/ChatPage'
import { LoadingState } from '../LoadingState/LoadingState'
import { ThinkingTrace } from '../ThinkingTrace/ThinkingTrace'

type MessageBubbleProps = {
  message: ChatMessage
  onRegenerate?: (assistantId: string) => void
}

const CopyIcon = () => (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="9" y="9" width="12" height="12" rx="2.5" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
)

const CheckIcon = () => (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 12.5l4.5 4.5L19 7" />
  </svg>
)

const RegenerateIcon = () => (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6" />
  </svg>
)

const ThumbUpIcon = () => (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M7 10v12M15 5.88L14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88z" />
  </svg>
)

const ThumbDownIcon = () => (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M17 14V2M9 18.12L10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88z" />
  </svg>
)

export function MessageBubble({ message, onRegenerate }: MessageBubbleProps) {
  const { id, role, content, sources, citations, trace, phase, startedAt, finishedAt } = message
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)

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

      {isDone && content ? (
        <div className="msg-actions">
          <button
            type="button"
            className="msg-action-icon"
            onClick={handleCopy}
            aria-label={copied ? 'Copied' : 'Copy answer'}
            title="Copy"
          >
            {copied ? <CheckIcon /> : <CopyIcon />}
          </button>

          {onRegenerate ? (
            <button
              type="button"
              className="msg-action-icon"
              onClick={() => onRegenerate(id)}
              aria-label="Regenerate answer"
              title="Regenerate"
            >
              <RegenerateIcon />
            </button>
          ) : null}

          <button
            type="button"
            className={`msg-action-icon ${feedback === 'up' ? 'active' : ''}`}
            onClick={() => setFeedback((value) => (value === 'up' ? null : 'up'))}
            aria-label="Good response"
            aria-pressed={feedback === 'up'}
            title="Good response"
          >
            <ThumbUpIcon />
          </button>

          <button
            type="button"
            className={`msg-action-icon ${feedback === 'down' ? 'active' : ''}`}
            onClick={() => setFeedback((value) => (value === 'down' ? null : 'down'))}
            aria-label="Bad response"
            aria-pressed={feedback === 'down'}
            title="Bad response"
          >
            <ThumbDownIcon />
          </button>

          {hasSources ? (
            <div className="source-stack" tabIndex={0}>
              <span className="source-stack-icons" aria-hidden="true">
                {sources!.slice(0, 4).map((source, index) => (
                  <span
                    key={source.chunk_id}
                    className="source-token"
                    style={{ zIndex: sources!.length - index }}
                  >
                    {source.reference_id ?? index + 1}
                  </span>
                ))}
                {sources!.length > 4 ? (
                  <span className="source-token source-token-more">
                    +{sources!.length - 4}
                  </span>
                ) : null}
              </span>
              <span className="source-stack-label">
                {sources!.length} {sources!.length === 1 ? 'source' : 'sources'}
              </span>

              <div className="source-popover" role="list">
                <span className="source-popover-title">Sources</span>
                <ul>
                  {sources!.map((source) => (
                    <li key={source.chunk_id} role="listitem">
                      {source.reference_id != null ? (
                        <span className="source-popover-index">{source.reference_id}</span>
                      ) : null}
                      <span className="source-popover-body">
                        <span className="source-popover-name">{source.title}</span>
                        {source.snippet ? (
                          <span className="source-popover-snippet">{source.snippet}</span>
                        ) : null}
                        <span className="source-popover-score">
                          score {source.score.toFixed(4)}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  )
}
