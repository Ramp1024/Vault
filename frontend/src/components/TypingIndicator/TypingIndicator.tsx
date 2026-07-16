export function TypingIndicator() {
  return (
    <div className="message-row assistant" aria-live="polite" aria-label="Assistant is typing">
      <div className="message-bubble typing-indicator">
        <span />
        <span />
        <span />
      </div>
    </div>
  )
}
