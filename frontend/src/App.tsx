import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { sendFeedback, streamChat, type Source } from './api'

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  pending?: boolean
  error?: boolean
  traceId?: string | null
  feedback?: boolean
}

let nextId = 0

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => () => abortRef.current?.abort(), [])

  const updateMessage = (id: number, patch: (m: Message) => Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)))
  }

  const send = async (e?: FormEvent) => {
    e?.preventDefault()
    const text = input.trim()
    if (!text || busy) return

    const assistantId = ++nextId
    setMessages((prev) => [
      ...prev,
      { id: ++nextId, role: 'user', content: text },
      { id: assistantId, role: 'assistant', content: '', pending: true },
    ])
    setInput('')
    setBusy(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamChat(
        text,
        {
          onToken: (token) => updateMessage(assistantId, (m) => ({ content: m.content + token })),
          onDone: ({ sources, error, trace_id }) =>
            updateMessage(assistantId, () => ({
              sources,
              pending: false,
              error: Boolean(error),
              traceId: trace_id,
            })),
        },
        controller.signal,
      )
    } catch (err) {
      if (!controller.signal.aborted) {
        console.error(err)
        updateMessage(assistantId, (m) => ({
          content: m.content || "I couldn't reach the server. Please try again.",
          error: true,
        }))
      }
    } finally {
      updateMessage(assistantId, () => ({ pending: false }))
      setBusy(false)
    }
  }

  const rate = async (message: Message, helpful: boolean) => {
    if (!message.traceId || message.feedback !== undefined) return
    updateMessage(message.id, () => ({ feedback: helpful }))
    try {
      await sendFeedback(message.traceId, helpful)
    } catch (err) {
      console.error(err)
      updateMessage(message.id, () => ({ feedback: undefined }))
    }
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Company Helper</h1>
        <p>Ask anything about our internal documentation.</p>
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <div className="empty">Try asking about a process, policy, or project.</div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`message ${m.role}${m.error ? ' error' : ''}`}>
            <div className="bubble">
              <AnswerText content={m.content} sources={m.sources} />
              {m.pending && !m.content && <span className="typing">Searching docs…</span>}
              {m.pending && m.content && <span className="cursor" />}
            </div>
            {m.sources && m.sources.length > 0 && <SourceList sources={m.sources} />}
            {m.traceId && !m.pending && (
              <div className="feedback">
                <button
                  type="button"
                  className={m.feedback === true ? 'selected' : ''}
                  disabled={m.feedback !== undefined}
                  onClick={() => rate(m, true)}
                  aria-label="Helpful"
                >
                  👍
                </button>
                <button
                  type="button"
                  className={m.feedback === false ? 'selected' : ''}
                  disabled={m.feedback !== undefined}
                  onClick={() => rate(m, false)}
                  aria-label="Not helpful"
                >
                  👎
                </button>
                {m.feedback !== undefined && <span>Thanks for the feedback</span>}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={send}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question…"
          rows={1}
          disabled={busy}
          autoFocus
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

const CITATION_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g

// Renders inline [1] / [1, 2] markers as citation badges, linked once sources arrive.
function AnswerText({ content, sources }: { content: string; sources?: Source[] }) {
  const parts: React.ReactNode[] = []
  let last = 0
  for (const match of content.matchAll(CITATION_RE)) {
    parts.push(content.slice(last, match.index))
    for (const n of match[1].split(',').map((x) => Number(x.trim()))) {
      const source = sources?.find((s) => s.id === n)
      parts.push(
        source ? (
          <a key={`${match.index}-${n}`} className="cite" href={source.url} target="_blank" rel="noreferrer" title={source.title}>
            {n}
          </a>
        ) : (
          <span key={`${match.index}-${n}`} className="cite">{n}</span>
        ),
      )
    }
    last = match.index + match[0].length
  }
  parts.push(content.slice(last))
  return <>{parts}</>
}

function SourceList({ sources }: { sources: Source[] }) {
  return (
    <div className="sources">
      {sources.map((s) => (
        <a key={s.url} className="source-card" href={s.url} target="_blank" rel="noreferrer">
          <span className="source-title">
            <span className="cite">{s.id}</span> {s.title}
          </span>
        </a>
      ))}
    </div>
  )
}

export default App
