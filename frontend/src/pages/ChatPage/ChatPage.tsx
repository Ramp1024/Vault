import { useCallback, useMemo, useState } from 'react'
import { ChatInput } from '../../components/ChatInput/ChatInput'
import { ChatWindow } from '../../components/ChatWindow/ChatWindow'

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

export type ChatMessage = {
    id: string
    role: ChatRole
    content: string
    sources?: ChatSource[]
    citations?: ChatCitation[]
}

const createMessage = (role: ChatRole, content: string): ChatMessage => ({
    id: `${role}-${crypto.randomUUID()}`,
    role,
    content,
})

export function ChatPage() {
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [isSending, setIsSending] = useState(false)

    const canSend = useMemo(() => !isSending, [isSending])

    const handleSend = useCallback(
        async (rawMessage: string) => {
            const message = rawMessage.trim()
            if (!message || isSending) {
                return
            }

            const userMessage = createMessage('user', message)
            const assistantMessage = createMessage('assistant', '')

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

                const applyContent = (text: string) => {
                    if (!text) {
                        return
                    }
                    setMessages((current) =>
                        current.map((entry) =>
                            entry.id === assistantMessage.id
                                ? { ...entry, content: entry.content + text }
                                : entry,
                        ),
                    )
                }

                const applySources = (sources: ChatSource[]) => {
                    setMessages((current) =>
                        current.map((entry) =>
                            entry.id === assistantMessage.id ? { ...entry, sources } : entry,
                        ),
                    )
                }

                const applyCitations = (citations: ChatCitation[]) => {
                    setMessages((current) =>
                        current.map((entry) =>
                            entry.id === assistantMessage.id ? { ...entry, citations } : entry,
                        ),
                    )
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
                            ? { ...entry, content: messageText }
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
                <h1>Vault Chat</h1>
                <p>Ask anything from your indexed Vault knowledge base.</p>
            </header>

            <ChatWindow messages={messages} isStreaming={isSending} />

            <ChatInput onSend={handleSend} disabled={!canSend} />
        </main>
    )
}
