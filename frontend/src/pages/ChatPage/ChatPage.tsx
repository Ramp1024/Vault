import { useCallback, useMemo, useState } from 'react'
import { ChatInput } from '../../components/ChatInput/ChatInput'
import { ChatWindow } from '../../components/ChatWindow/ChatWindow'
import { ThemeToggle } from '../../components/ThemeToggle/ThemeToggle'
import { useTheme } from '../../hooks/useTheme'

export type ChatRole = 'user' | 'assistant'

export type ChatSource = {
    reference_id?: number
    chunk_id: string
    document_id: string
    title: string
    score: number
    snippet: string
}

export type ChatCitation = {
    reference_id: number
    document_id: string
    chunk_id: string
    title: string
}

export type TraceStep = {
    id: string
    label: string
    status: 'active' | 'done'
}

export type ChatPhase = 'loading' | 'streaming' | 'done'

export type ChatMessage = {
    id: string
    role: ChatRole
    content: string
    sources?: ChatSource[]
    citations?: ChatCitation[]
    phase?: ChatPhase
    trace?: TraceStep[]
    startedAt?: number
    finishedAt?: number
}

const createMessage = (role: ChatRole, content: string): ChatMessage => ({
    id: `${role}-${crypto.randomUUID()}`,
    role,
    content,
})

// Build the trace shown once retrieval finishes. Grounded in the real stream:
// the sources header tells us the vault was searched and how many chunks matched.
const buildRetrievalTrace = (sourceCount: number): TraceStep[] => [
    { id: 'analyze', label: 'Understanding your question', status: 'done' },
    { id: 'search', label: 'Searching your Vault', status: 'done' },
    {
        id: 'read',
        label:
            sourceCount > 0
                ? `Reviewed ${sourceCount} matched ${sourceCount === 1 ? 'source' : 'sources'}`
                : 'No matching sources found',
        status: 'done',
    },
    { id: 'compose', label: 'Writing the answer', status: 'active' },
]

// Mark the composing step active once answer text begins to stream. Falls back to
// a minimal trace if the sources header never arrived.
const markComposing = (trace: TraceStep[] | undefined): TraceStep[] => {
    if (!trace || trace.length === 0) {
        return [
            { id: 'analyze', label: 'Understanding your question', status: 'done' },
            { id: 'compose', label: 'Writing the answer', status: 'active' },
        ]
    }
    if (trace.some((step) => step.id === 'compose')) {
        return trace
    }
    return [
        ...trace.map((step) => ({ ...step, status: 'done' as const })),
        { id: 'compose', label: 'Writing the answer', status: 'active' },
    ]
}

// Close out the trace: every step is done and the timer is frozen.
const finalizeTrace = (trace: TraceStep[] | undefined): TraceStep[] | undefined =>
    trace?.map((step) => ({ ...step, status: 'done' as const }))

export function ChatPage() {
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [isSending, setIsSending] = useState(false)
    const { theme, setTheme } = useTheme()

    const canSend = useMemo(() => !isSending, [isSending])

    const handleSend = useCallback(
        async (rawMessage: string) => {
            const message = rawMessage.trim()
            if (!message || isSending) {
                return
            }

            const userMessage = createMessage('user', message)
            const assistantMessage: ChatMessage = {
                ...createMessage('assistant', ''),
                phase: 'loading',
                startedAt: Date.now(),
                trace: [
                    {
                        id: 'analyze',
                        label: 'Understanding your question',
                        status: 'active',
                    },
                ],
            }

            setMessages((current) => [...current, userMessage, assistantMessage])
            setIsSending(true)

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message }),
                })

                if (!response.ok) {
                    throw new Error(`Chat request failed with status ${response.status}`)
                }

                if (!response.body) {
                    throw new Error('Server returned no stream body')
                }

                // getReader - Returns Reader object to pull streamed data from the response body
                const reader = response.body.getReader()

                // TextDecoder - Built in class for converting a stream of bytes into a string
                const decoder = new TextDecoder()

                // The stream begins with a single JSON header line describing the
                // retrieved sources, followed by '\n', then the raw answer text,
                // and finally a NUL-delimited JSON metadata frame carrying the
                // backend-validated citations. The rendered answer is exactly the
                // text between the header newline and the NUL delimiter.
                const FINAL_FRAME_DELIMITER = '\u0000'
                let headerParsed = false
                let headerBuffer = ''
                let inFinalFrame = false
                let finalBuffer = ''
                let streamingStarted = false

                const patchAssistant = (
                    patch: Partial<ChatMessage> | ((entry: ChatMessage) => Partial<ChatMessage>),
                ) => {
                    setMessages((current) =>
                        current.map((entry) =>
                            entry.id === assistantMessage.id
                                ? {
                                      ...entry,
                                      ...(typeof patch === 'function' ? patch(entry) : patch),
                                  }
                                : entry,
                        ),
                    )
                }

                const applyContent = (text: string) => {
                    if (!text) {
                        return
                    }
                    if (!streamingStarted) {
                        streamingStarted = true
                        patchAssistant((entry) => ({
                            content: entry.content + text,
                            phase: 'streaming',
                            trace: markComposing(entry.trace),
                        }))
                        return
                    }
                    patchAssistant((entry) => ({ content: entry.content + text }))
                }

                const applySources = (sources: ChatSource[]) => {
                    patchAssistant({ sources, trace: buildRetrievalTrace(sources.length) })
                }

                const applyCitations = (citations: ChatCitation[]) => {
                    patchAssistant({ citations })
                }

                // Route answer-body text to the message, splitting off the trailing
                // metadata frame once the NUL delimiter appears.
                const handleBody = (text: string) => {
                    if (inFinalFrame) {
                        finalBuffer += text
                        return
                    }
                    const delimiterIndex = text.indexOf(FINAL_FRAME_DELIMITER)
                    if (delimiterIndex === -1) {
                        applyContent(text)
                        return
                    }
                    applyContent(text.slice(0, delimiterIndex))
                    inFinalFrame = true
                    finalBuffer += text.slice(delimiterIndex + 1)
                }

                const handlePiece = (piece: string) => {
                    if (!piece) {
                        return
                    }
                    if (headerParsed) {
                        handleBody(piece)
                        return
                    }

                    headerBuffer += piece
                    const newlineIndex = headerBuffer.indexOf('\n')
                    if (newlineIndex === -1) {
                        return
                    }

                    const headerText = headerBuffer.slice(0, newlineIndex)
                    const rest = headerBuffer.slice(newlineIndex + 1)
                    headerParsed = true

                    try {
                        const parsed = JSON.parse(headerText) as { sources?: ChatSource[] }
                        if (parsed.sources) {
                            applySources(parsed.sources)
                        }
                    } catch {
                        // If the header can't be parsed, keep the text so nothing is lost.
                        applyContent(headerText)
                    }
                    handleBody(rest)
                }

                // Parse the accumulated metadata frame once the stream ends. The
                // answer text is already rendered, so a malformed frame is
                // non-fatal — validated citations are simply omitted.
                const flushFinalFrame = () => {
                    if (!finalBuffer) {
                        return
                    }
                    try {
                        const parsed = JSON.parse(finalBuffer.trim()) as {
                            type?: string
                            citations?: ChatCitation[]
                        }
                        if (parsed.citations) {
                            applyCitations(parsed.citations)
                        }
                    } catch {
                        // Ignore malformed trailing metadata.
                    }
                }

                try {
                    while (true) {
                        const { value, done } = await reader.read()
                        if (done) {
                            break
                        }

                        handlePiece(decoder.decode(value, { stream: true }))
                    }

                    handlePiece(decoder.decode())

                    // Guard: if no header newline ever arrived, don't lose the text.
                    if (!headerParsed && headerBuffer) {
                        applyContent(headerBuffer)
                    }

                    flushFinalFrame()

                    patchAssistant((entry) => ({
                        phase: 'done',
                        finishedAt: Date.now(),
                        trace: finalizeTrace(entry.trace),
                    }))
                } finally {
                    reader.releaseLock()
                }
            } catch (error) {
                const messageText =
                    error instanceof Error
                        ? `Unable to generate response: ${error.message}`
                        : 'Unable to generate response.'

                setMessages((current) =>
                    current.map((entry) =>
                        entry.id === assistantMessage.id
                            ? {
                                  ...entry,
                                  content: messageText,
                                  phase: 'done',
                                  finishedAt: Date.now(),
                                  trace: undefined,
                              }
                            : entry,
                    ),
                )
            } finally {
                setIsSending(false)
            }
        },
        [isSending],
    )

    return (
        <main className="chat-page">
            <header className="chat-header">
                <div className="chat-brand">
                    <span className="chat-brand-mark" aria-hidden="true">V</span>
                    <div className="chat-brand-text">
                        <h1>
                            Vault <span className="chat-brand-badge">Chat</span>
                        </h1>
                        <p>Answers grounded in your indexed knowledge base.</p>
                    </div>
                </div>
                <ThemeToggle theme={theme} onSelect={setTheme} />
            </header>

            <ChatWindow
                messages={messages}
                isStreaming={isSending}
                onPromptSelect={handleSend}
            />

            <ChatInput onSend={handleSend} disabled={!canSend} />
        </main>
    )
}
