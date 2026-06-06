/* ============================================================
   SSE chat stream — POST /v1/chat/completions (and /confirm reuse).
   The engine adapter discards any client message array and rebuilds
   history from the DB; we only send `derpr_user_text` + flags + params.

   We use fetch + ReadableStream (not EventSource, which is GET-only) and
   parse the SSE grammar in order:
     data: {chat.completion.chunk}        → token delta
     event: derpr-tool-start              → live tool card
     event: derpr-tool-result             → tool result
     event: derpr-confirm                 → parked write (S6; surfaced as event)
     event: derpr  data:{id-frame}        → hydrate ids/response_type (EVERY terminal turn)
     data: {"error":{...}}                → error frame
     data: [DONE]                         → finalize
   ============================================================ */
import type {
  DerprIdFrame,
  ToolStartFrame,
  ToolResultFrame,
} from '../types/contracts'

export interface ConfirmFrame {
  text: string
  persona: string
  token: string
  calls: { name: string; arguments: Record<string, unknown>; id: string }[]
  audit_info?: unknown
}

export interface StreamHandlers {
  onToken: (delta: string) => void
  onToolStart: (f: ToolStartFrame) => void
  onToolResult: (f: ToolResultFrame) => void
  onConfirm?: (f: ConfirmFrame) => void
  onIdFrame: (f: DerprIdFrame) => void
  onError: (message: string) => void
  onDone: () => void
}

export interface ChatRequest {
  derpr_user_text: string
  derpr_retry?: boolean
  model?: string
  // DP-136 6b: scope the turn to a channel (defaults to web_ui server-side).
  // The engine logs the turn under this channel and rebuilds history per the
  // persona's memory_mode, so submitting with a fresh tag "creates" a channel.
  channel?: string
  user_identifier?: string
  server_id?: string
  temperature?: number
  top_p?: number
  top_k?: number
  max_tokens?: number
  stop?: string[] | null
  rep_pen?: number
  min_p?: number
  tfs?: number
}

// Parse one SSE block ("event:" + "data:" lines) and dispatch.
function dispatchBlock(block: string, h: StreamHandlers, sawError: { v: boolean }) {
  const lines = block.split('\n')
  let event = 'message'
  const dataLines: string[] = []
  for (const line of lines) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''))
  }
  const data = dataLines.join('\n')
  if (!data) return

  if (data === '[DONE]') {
    h.onDone()
    return
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(data)
  } catch {
    return
  }
  const obj = parsed as Record<string, unknown>

  switch (event) {
    case 'derpr-tool-start':
      h.onToolStart(obj as unknown as ToolStartFrame)
      return
    case 'derpr-tool-result':
      h.onToolResult(obj as unknown as ToolResultFrame)
      return
    case 'derpr-confirm':
      h.onConfirm?.(obj as unknown as ConfirmFrame)
      return
    case 'derpr':
      h.onIdFrame(obj as unknown as DerprIdFrame)
      return
    default: {
      // unlabelled `data:` — either a token delta or an error frame
      if (obj.error) {
        sawError.v = true
        const e = obj.error as { message?: string }
        h.onError(e.message || 'engine error')
        return
      }
      const choices = obj.choices as { delta?: { content?: string } }[] | undefined
      const delta = choices?.[0]?.delta?.content
      if (typeof delta === 'string' && delta.length) h.onToken(delta)
    }
  }
}

async function consume(
  resp: Response,
  h: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  if (!resp.body) {
    h.onError('no response body')
    return
  }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  const sawError = { v: false }
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      if (signal.aborted) break
      buf += decoder.decode(value, { stream: true })
      // SSE blocks are separated by a blank line.
      let idx: number
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const block = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        dispatchBlock(block, h, sawError)
      }
    }
    if (buf.trim()) dispatchBlock(buf, h, sawError)
  } catch (e) {
    if (!signal.aborted) h.onError(String(e))
  }
}

/** Open the chat stream. Returns an abort handle. */
export function streamChat(req: ChatRequest, h: StreamHandlers): { abort: () => void } {
  const ctrl = new AbortController()
  fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ stream: true, ...req }),
    signal: ctrl.signal,
  })
    .then((resp) => {
      if (!resp.ok) {
        h.onError(`chat → ${resp.status}`)
        return
      }
      return consume(resp, h, ctrl.signal)
    })
    .catch((e) => {
      if (!ctrl.signal.aborted) h.onError(String(e))
    })
  return { abort: () => ctrl.abort() }
}

/** Resolve a parked CONFIRM write (S6 reuse). Same wire protocol. */
export function streamConfirm(
  persona: string,
  approved: boolean,
  token: string,
  h: StreamHandlers,
): { abort: () => void } {
  const ctrl = new AbortController()
  fetch(`/api/v1/persona/${encodeURIComponent(persona)}/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ approved, token }),
    signal: ctrl.signal,
  })
    .then((resp) => {
      if (!resp.ok) {
        h.onError(`confirm → ${resp.status}`)
        return
      }
      return consume(resp, h, ctrl.signal)
    })
    .catch((e) => {
      if (!ctrl.signal.aborted) h.onError(String(e))
    })
  return { abort: () => ctrl.abort() }
}
