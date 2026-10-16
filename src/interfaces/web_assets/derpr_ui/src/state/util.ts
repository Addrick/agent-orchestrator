/* shared helpers — kept dumb and pure */
import type { Chunk } from '../types/contracts'

/** A chunk's reasoning and reply. Since DP-434 reasoning arrives in its own
 *  field; rows persisted before that carry it folded into content as a
 *  leading '<think>\nR\n</think>\nC' block with `reasoning` null, so split
 *  those here. Temporary: remove once no legacy rows remain. */
export function chunkParts(c: Chunk): { reasoning: string | null; body: string } {
  if (c.role !== 'assistant') return { reasoning: null, body: c.content }
  if (c.reasoning) return { reasoning: c.reasoning, body: c.content }
  const m = c.content.match(/^<think>\s*([\s\S]*?)\s*<\/think>\s*([\s\S]*)$/)
  if (m) return { reasoning: m[1], body: m[2] }
  return { reasoning: null, body: c.content }
}

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
