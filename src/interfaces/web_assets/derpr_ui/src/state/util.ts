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
// reply. Mirrors the engine's `split_think_text`.
export function joinRawTurn(reasoning: string | null | undefined, reply: string): string {
  return reasoning ? `<think>\n${reasoning}\n</think>\n\n${reply}` : reply
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
