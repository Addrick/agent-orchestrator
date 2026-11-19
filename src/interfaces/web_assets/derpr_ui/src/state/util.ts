/* shared helpers — kept dumb and pure */

export function fmtTok(n: number): string {
  return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n)
}

/** Display label for a tool_policy. The live engine omits `mode` and exposes
 *  `default` ('deny'|'allow'|'ask') instead; fall back to that. */
export function policyLabel(
  tp?: { mode?: string; default?: string } | null,
): string {
  if (!tp) return '—'
  if (tp.mode) return tp.mode
  if (tp.default) return tp.default.toUpperCase()
  return '—'
}

/** DP-510: a re-fetched transcript arrives as all-new objects, which defeats
 *  MessageRow's memo and re-parses every row's markdown/KaTeX/tool cards — on
 *  the 5 s codr poll and every window focus. Keep the previous object for any
 *  row whose data is unchanged, and the previous array when nothing changed. */
export function reuseChunks<T extends { interaction_id: number | null; ephemeral_chunk_id?: string }>(
  prev: T[],
  next: T[],
): T[] {
  const key = (c: T) =>
    c.interaction_id != null ? `i${c.interaction_id}` : c.ephemeral_chunk_id ? `e${c.ephemeral_chunk_id}` : null
  const byKey = new Map<string, T>()
  for (const c of prev) {
    const k = key(c)
    if (k) byKey.set(k, c)
  }
  let same = prev.length === next.length
  const out = next.map((c, i) => {
    const k = key(c)
    const old = k ? byKey.get(k) : undefined
    const kept = old && JSON.stringify(old) === JSON.stringify(c) ? old : c
    if (kept !== prev[i]) same = false
    return kept
  })
  return same ? prev : out
}

/** DP-526: a saved time, compact — `14:05` today, `Mon 14:05` within the
 *  week, `Nov 3` this year, `2025-11-03` before. Browser time zone. '' when
 *  absent or unparseable. */
export function fmtWhen(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (isNaN(d.getTime())) return ''
  const hm = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  if (d >= startOfToday) return hm
  if (now.getTime() - d.getTime() < 6 * 86400000)
    return `${d.toLocaleDateString([], { weekday: 'short' })} ${hm}`
  if (d.getFullYear() === now.getFullYear())
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

/** DP-526: the full saved time, for a hover title. */
export function fmtWhenFull(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  return isNaN(d.getTime()) ? '' : d.toLocaleString([], { dateStyle: 'full', timeStyle: 'medium' })
}

/** Cheap client-side estimate — DEMO/budget-only. Production token counts must
 *  come from POST /api/extra/tokencount or the /assemble payload (S4/S5). */
export function estimateTokens(s: string): number {
  return Math.round((s || '').length / 3.6)
}

export function argString(args: Record<string, unknown>): string {
  return Object.entries(args || {})
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(', ')
}

export type Treatment =
  | 'normal'
  | 'tool-only'
  | 'parked'
  | 'aborted'
  | 'error'
  | 'security-blocked'

/** Derive a response-type treatment from the wire id-frame + stream signals.
 *  The engine's ResponseType enum is coarse (LLM_GENERATION / DEV_COMMAND /
 *  PENDING_CONFIRMATION); the README's six descriptive treatments are inferred
 *  from that plus whether the turn parked, was aborted, errored, or ran tools
 *  with empty content. */
export function deriveTreatment(opts: {
  responseType?: string
  ephemeralChunkId?: string | null
  aborted?: boolean
  errored?: boolean
  hadTools?: boolean
  emptyContent?: boolean
  securityBlocked?: boolean
}): Treatment {
  if (opts.securityBlocked) return 'security-blocked'
  if (opts.errored) return 'error'
  if (opts.aborted) return 'aborted'
  const rt = (opts.responseType || '').toUpperCase()
  if (rt === 'PENDING_CONFIRMATION' || rt === 'PARKED' || opts.ephemeralChunkId)
    return 'parked'
  if (opts.hadTools && opts.emptyContent) return 'tool-only'
  return 'normal'
}

// DP-447 raw turn edits: one text holding `<think>` + reasoning + `</think>` +
// reply. Mirrors the engine's `split_think_text`. Thinking with no reply yet
// is left open, as the engine reads that row (`render_continue_prefix`): a
// closing tag here would make an unchanged save mark it closed, and continue
// would then answer instead of carrying on the thought.
export function joinRawTurn(reasoning: string | null | undefined, reply: string): string {
  if (!reasoning) return reply
  return reply ? `<think>\n${reasoning}\n</think>\n\n${reply}` : `<think>\n${reasoning}`
}

// How a raw edit leaves the thinking: 'open' (no `</think>` — continue keeps
// thinking), 'closed-empty' (closed, no reply yet — the one shape the stored
// columns lose), 'closed', or null when the text has no think block.
export function rawThinkState(raw: string): 'open' | 'closed-empty' | 'closed' | null {
  const t = raw.trimStart()
  if (!t.startsWith('<think>')) return null
  const close = t.indexOf('</think>')
  if (close === -1) return 'open'
  return t.slice(close + '</think>'.length).trim() ? 'closed' : 'closed-empty'
}
