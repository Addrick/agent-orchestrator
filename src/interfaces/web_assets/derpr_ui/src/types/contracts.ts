/* ============================================================
   DERPR Portal — engine-adapter contract types.
   Mirrors API_CONTRACTS.md and design/portal-data.js exactly.
   Chunks are addressed by interaction_id / ephemeral_chunk_id,
   NEVER by array position (invariants C1/C2).
   ============================================================ */

// ---- GET /api/v1/session/{persona}/transcript → { chunks: [...] } ----
export interface ToolContext {
  call_id: string
  group_id: string | null
  tool_name: string
  arguments: Record<string, unknown>
  result: string | null
  error: string | null
}

export interface Chunk {
  interaction_id: number | null // null only when ephemeral
  role: 'user' | 'assistant'
  content: string // reasoning folded in as <think>…</think>
  ephemeral: boolean
  reasoning: string | null
  tool_context: ToolContext[] | null
  has_versions: boolean
  ephemeral_chunk_id?: string // present ONLY on the parked ephemeral chunk
}

export interface TranscriptResponse {
  chunks: Chunk[]
}

// ---- GET /api/v1/persona/{name} ----
export interface ToolPolicy {
  // The live engine returns {default, allow, ask, ...}; there is NO `mode`
  // field (the handoff contract was wrong about that). Keep `mode` optional
  // for forward-compat and derive a display label from `default` otherwise.
  mode?: string
  default?: string // 'deny' | 'allow' | 'ask'
  allow?: string[]
  ask?: string[]
  // Security-invariant escape hatches (policy.py validate_composition) and
  // any capability gates — preserved on round-trip through the editor.
  explicit_overrides?: string[]
  capabilities_required?: string[]
  [k: string]: unknown
}

export interface KoboldExtras {
  rep_pen?: number
  rep_pen_range?: number
  rep_pen_slope?: number
  min_p?: number
  typical?: number
  tfs?: number
  mirostat?: number
  mirostat_tau?: number
  mirostat_eta?: number
  sampler_order?: number[]
  [k: string]: unknown
}

export interface Persona {
  name: string
  display_name: string
  prompt: string
  // base params
  model_name: string
  // Optional base params are null on personas that don't set them (the live
  // engine returns null, unlike the fully-populated mock).
  temperature: number | null
  max_tokens: number
  history_messages: number
  thinking_level: string | null
  memory_mode: string
  max_context_tokens: number
  chat_template: string | null
  tool_policy: ToolPolicy | null
  enabled_tools: string[]
  // kobold-only
  top_p: number | null
  top_k: number | null
  instruct_tags: Record<string, string> | null
  kobold_extras: KoboldExtras
  // security
  security_blocked: boolean
  security_block_reasons: string[]
}

export interface PatchPersonaResult {
  result: string
  rejected_fields: string[]
  unknown_fields: string[]
  error?: string
  detail?: string
}

// ---- GET /api/v1/tools/catalog ----
export interface ToolCapabilities {
  locality: 'local' | 'remote'
  sensitivity: 'low' | 'medium' | 'high'
  produces_untrusted: boolean
}

export interface ToolDef {
  name: string
  description: string
  is_write: boolean
  // owning service (e.g. 'zammad'); null/absent for built-in/local tools.
  service_binding?: string | null
  capabilities: ToolCapabilities
}

export interface ToolsCatalog {
  tools: ToolDef[]
}

// ---- GET /api/v1/session/{persona}/ltm_block ----
export interface LtmBlockResponse {
  block: string | null
}

// ---- channel tags (source-agnostic `channel` string) ----
export type ChannelSource = 'web' | 'dsc' | 'zmd' | 'gml'

// ---- GET /api/v1/channels → { channels: ChannelRow[] } (DP-136 6b) ----
export interface ChannelRow {
  channel: string
  server_id: string | null
  source: ChannelSource
  count: number
  last_ts: string | null
}

export interface ChannelItem {
  id: string
  // the raw source-agnostic `channel` string used to scope transcript/submit
  channel: string
  name: string
  source: ChannelSource
  persona: string
  active?: boolean
  preview: string
}
export interface ChannelGroup {
  group: string
  items: ChannelItem[]
}

// ---- response types (DoneEvent.response_type) ----
export type ResponseType =
  | 'NORMAL'
  | 'TOOL_ONLY'
  | 'PARKED'
  | 'ABORTED'
  | 'ERROR'
  | 'SECURITY_BLOCKED'

// ---- SSE id-frame (DoneEvent) ----
export interface DerprIdFrame {
  user_id: number | null
  assistant_id: number | null
  response_type: string
  ephemeral_chunk_id: string | null
}

// ---- SSE tool frames ----
export interface ToolStartFrame {
  tool_name: string
  arguments: Record<string, unknown>
  call_id: string
  group_id: string | null
}
export interface ToolResultFrame {
  call_id: string
  tool_name: string
  result: string | null
  error: string | null
  group_id: string | null
}

// ---- versions (GET /interaction/{id}/versions) ----
export interface VersionEntry {
  version?: number
  content: string
  reasoning?: string | null
  canonical?: boolean
  [k: string]: unknown
}
export interface VersionsResponse {
  interaction_id: number
  versions: VersionEntry[] // canonical LAST
}

// ---- dev command ----
export interface DevCommandResponse {
  response: string
  mutated?: boolean
}

// ---- GET /api/v1/session/{persona}/assemble (S5 parity inspector) ----
export interface AssembledMessage {
  role: string
  content: string
  // provenance tag: persona.prompt | ltm_block | history | composer |
  // tool_call | tool_result — maps the wire line back to its source row.
  src: string
}
export interface AssembledParity {
  // engine.dry_run = produced by the shared live builder (green banner);
  // client_fallback = reconstructed in the browser, may drift (red banner).
  source: 'engine.dry_run' | 'client_fallback'
  builder: string
  matches_live: boolean
}
export interface AssembledRequest {
  parity: AssembledParity
  route: string
  model_name: string
  // flattened resolved params (universal + kobold extras forwarded for the route)
  params: Record<string, unknown>
  messages: AssembledMessage[]
}
