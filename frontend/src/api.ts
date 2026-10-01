export interface Source {
  id: number
  title: string
  url: string
  score: number
}

export interface DonePayload {
  sources: Source[]
  error: string | null
}

interface StreamHandlers {
  onToken: (token: string) => void
  onDone: (payload: DonePayload) => void
}

// EventSource only supports GET, so the SSE stream from POST /chat is parsed by hand.
export async function streamChat(
  message: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new Error(`Request failed with status ${res.status}`)
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value

    let boundary: number
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      dispatch(rawEvent, handlers)
    }
  }
}

function dispatch(rawEvent: string, handlers: StreamHandlers) {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of rawEvent.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  if (dataLines.length === 0) return

  const data = JSON.parse(dataLines.join('\n'))
  if (event === 'token') handlers.onToken(data)
  else if (event === 'done') handlers.onDone(data)
}
