/* ============================================================
   SSE chat stream — POST /v1/chat/completions (and /confirm reuse).
   The engine adapter discards any client message array and rebuilds
   history from the DB; we only send `derpr_user_text` + flags + params.

   We use fetch + ReadableStream (not EventSource, which is GET-only) and
   parse the SSE grammar in order:
     data: {chat.completion.chunk}        → token delta (delta.content) or
                                            reasoning delta (delta.reasoning_content)
     event: derpr-tool-start              → live tool card
     event: derpr-tool-result             → tool result
     event: derpr-confirm                 → parked write (S6; surfaced as event)
     event: derpr  data:{id-frame}        → hydrate ids/response_type (EVERY terminal turn)
     data: {"error":{...}}                → error frame
     data: [DONE]                         → finalize
   ============================================================ */
import { withAuth } from './control_token'
import type {
  DerprIdFrame,
  ToolStartFrame,
  ToolResultFrame,
} from '../types/contracts'

export interface StreamHandlers {
  onToken: (delta: string) => void
  onReasoning: (delta: string) => void
  onToolStart: (f: ToolStartFrame) => void
  onToolResult: (f: ToolResultFrame) => void
  onIdFrame: (f: DerprIdFrame) => void
  onError: (message: string) => void
  onDone: () => void
  // DP-447: the engine refused the turn because the conversation moved on
  // since this page loaded it (HTTP 409 stale_view). Nothing was written.
  onStale?: () => void
}

export interface ChatRequest {
  derpr_user_text: string
  derpr_retry?: boolean
  // DP-447: extend the trailing assistant reply in place. `derpr_think_open`
  // says whether its thinking is still open when the row alone can't tell.
  derpr_continue?: boolean
  derpr_think_open?: boolean
  // The row being continued; the engine refuses if it would extend another.
  derpr_continue_id?: number
  // DP-447 stale-view guard: the interaction_id this screen ends on (null for
  // an empty conversation). The engine refuses the turn with 409 on mismatch.
  derpr_expected_last_id?: number | null
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
      // Deliberately ignored: parked writes are surfaced by the /transcript
      // re-sync after [DONE], not consumed off the wire.
      return
    case 'derpr':
      h.onIdFrame(obj as unknown as DerprIdFrame)
      return
    default: {
      // unlabelled `data:` — a token or reasoning delta, or an error frame
      if (obj.error) {
        sawError.v = true
        const e = obj.error as { message?: string }
        h.onError(e.message || 'engine error')
        return
      }
      const choices = obj.choices as
        | { delta?: { content?: string; reasoning_content?: string } }[]
        | undefined
      const d = choices?.[0]?.delta
      // DP-434: reasoning arrives in its own field, never inside content.
      if (typeof d?.reasoning_content === 'string' && d.reasoning_content.length)
        h.onReasoning(d.reasoning_content)
      if (typeof d?.content === 'string' && d.content.length) h.onToken(d.content)
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
        // An error frame terminates the turn client-side. Keep reading and a
        // later [DONE] on this same (still-open) connection would fire onDone
        // against whatever stream is CURRENT by then — clobbering a newly
        // started stream's state (handlers are not generation-tagged).
        if (sawError.v) {
          void reader.cancel().catch(() => {})
          return
        }
      }
    }
    if (buf.trim()) dispatchBlock(buf, h, sawError)
  } catch (e) {
    if (!signal.aborted) h.onError(String(e))
  }
}

// Shared POST-SSE scaffold: fetch with abort handle, HTTP-status check,
// then hand the body to `consume`.
function openSse(
  url: string,
  body: Record<string, unknown>,
  label: string,
  h: StreamHandlers,
): { abort: () => void } {
  const ctrl = new AbortController()
  fetch(url, {
    method: 'POST',
    // DP-277: attach the operator token when set. Required for /confirm (a
    // control-plane route, 401 without it) and it elevates chat so typed dev
    // commands work; anonymous chat still streams (data plane).
    headers: withAuth({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
    body: JSON.stringify(body),
    signal: ctrl.signal,
  })
    .then((resp) => {
      if (resp.status === 409 && h.onStale) {
        h.onStale()
        return
      }
      if (!resp.ok) {
        h.onError(`${label} → ${resp.status}`)
        return
      }
      return consume(resp, h, ctrl.signal)
    })
    .catch((e) => {
      if (!ctrl.signal.aborted) h.onError(String(e))
    })
  return { abort: () => ctrl.abort() }
}

/** Open the chat stream. Returns an abort handle. */
export function streamChat(req: ChatRequest, h: StreamHandlers): { abort: () => void } {
  return openSse('/v1/chat/completions', { stream: true, ...req }, 'chat', h)
}

/** Resolve a parked CONFIRM write (S6 reuse). Same wire protocol. */
export function streamConfirm(
  persona: string,
  approved: boolean,
  token: string,
  h: StreamHandlers,
): { abort: () => void } {
  return openSse(
    `/api/v1/persona/${encodeURIComponent(persona)}/confirm`,
    { approved, token },
    'confirm',
    h,
  )
}
