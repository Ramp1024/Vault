export function TypingIndicator() {
  return (
    <div className="msg msg-assistant" aria-live="polite" aria-label="Vault is thinking">
      <div className="msg-role">
        <span className="msg-role-dot" aria-hidden="true" />
        Vault
      </div>
      <div className="typing-indicator">
        <span />
        <span />
        <span />
      </div>
    </div>
  )
}
