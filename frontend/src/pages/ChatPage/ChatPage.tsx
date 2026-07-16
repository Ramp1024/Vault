import { useCallback, useMemo, useState } from 'react'
import { ChatInput } from '../../components/ChatInput/ChatInput'
import { ChatWindow } from '../../components/ChatWindow/ChatWindow'

export type ChatRole = 'user' | 'assistant'

export type ChatMessage = {
    id: string
    role: ChatRole
    content: string
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

                try {
                    while (true) {
                        const { value, done } = await reader.read()
                        if (done) {
                            break
                        }

                        const chunk = decoder.decode(value, { stream: true })
                        if (!chunk) {
                            continue
                        }

                        setMessages((current) =>
                            current.map((entry) =>
                                entry.id === assistantMessage.id
                                    ? { ...entry, content: entry.content + chunk }
                                    : entry,
                            ),
                        )
                    }

                    const finalChunk = decoder.decode()
                    if (finalChunk) {
                        setMessages((current) =>
                            current.map((entry) =>
                                entry.id === assistantMessage.id
                                    ? { ...entry, content: entry.content + finalChunk }
                                    : entry,
                            ),
                        )
                    }
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
