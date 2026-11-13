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
  // The thinking that led to this call, on the first card of each step
  // (DP-437). Rendered just above the card, in the order the model saw it.
  reasoning?: string | null
  // The text the model wrote beside this step's calls (DP-338), on the
  // step's first card (DP-440). The model reads it back on later turns.
  prose?: string | null
}

export interface Chunk {
  interaction_id: number | null // null only when ephemeral
  role: 'user' | 'assistant'
  content: string // the reply alone; reasoning is its own field (DP-434)
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
  // Resolved route ("default" sentinel expanded). Optional: older engines
  // omit it. Gates CONTINUE (DP-447), which only `local` supports.
  effective_model_name?: string
  // Optional base params are null on personas that don't set them (the live
  // engine returns null, unlike the fully-populated mock).
  temperature: number | null
  max_tokens: number
  history_messages: number
  thinking_level: string | null
  memory_mode: string
  long_term_memory: boolean
  max_context_tokens: number
  // per-persona tool-call budget; null = global MAX_TOOL_CALLS (DP-428)
  max_tool_calls: number | null
  // DP-491 LTM recall knobs; null = default (MEMORY_MAX_SUMMARIES_IN_CONTEXT /
  // Hindsight's own). ltm_max_tokens + ltm_budget are Hindsight-only.
  ltm_max_items?: number | null
  ltm_max_tokens?: number | null
  ltm_budget?: string | null
  chat_template: string | null
  tool_policy: ToolPolicy | null
  enabled_tools: string[]
  // services this persona may use tools from (gate 2 in _filter_tools_for_persona).
  // a service tool is only offered to the model if its binding is listed here.
  service_bindings: string[]
  // kobold-only
  top_p: number | null
  top_k: number | null
  kobold_extras: KoboldExtras
  // security
  security_blocked: boolean
  security_block_reasons: string[]
  // DP-513: PATCH key -> accepted values in words (shown next to the input).
  // Optional: older engines omit it.
  field_constraints?: Record<string, string>
}

export interface PatchPersonaResult {
  result: string
  rejected_fields: string[]
  // DP-513: rejected key -> the rule its value broke. Optional: older engines.
  rejected_reasons?: Record<string, string>
  unknown_fields: string[]
  error?: string
  detail?: string
}

// ---- POST /api/v1/personas (create) ----
export interface CreatePersonaResult {
  result: string // 'created'
  rejected_fields: string[]
  rejected_reasons?: Record<string, string>
  unknown_fields: string[]
  // the created persona, same shape as GET /persona/{name}
  persona: Persona
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
  // DP-497: setting an irreversible tool to allow asks the operator first.
  irreversible?: boolean
  // DP-497: the state a tool gets when enabled without one ("on"):
  // reads allow, writes and irreversible tools ask.
  default_state?: 'allow' | 'ask'
  // owning service (e.g. 'zammad'); null/absent for built-in/local tools.
  service_binding?: string | null
  capabilities: ToolCapabilities
}

export interface ToolsCatalog {
  tools: ToolDef[]
}

// ---- POST /api/v1/session/{persona}/ltm_block (JSON body, DP-500) ----
export interface LtmBlockResponse {
  block: string | null
  // DP-492: a newer draft preview for the same persona+channel cancelled
  // this one's recall; `block` is null and must not replace the panel.
  superseded?: boolean
}

// ---- channel tags (source-agnostic `channel` string) ----
// `tsk`: a codr ticket's channel, `task:<ticket>` (DP-484)
export type ChannelSource = 'web' | 'dsc' | 'zmd' | 'gml' | 'tsk'

// ---- GET /api/v1/channels → { channels: ChannelRow[] } (DP-136 6b) ----
export interface ChannelRow {
  channel: string
  server_id: string | null
  source: ChannelSource
  count: number
  last_ts: string | null
  // display name when the tag alone is not one (a ticket's `CD-001 · title`)
  label?: string | null
}

export interface ChannelItem {
  id: string
  // the raw source-agnostic `channel` string used to scope transcript/submit
  channel: string
  // what the rail shows; the raw tag when the engine sends no label
  label: string
  source: ChannelSource
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
// DP-487: a turn running on the server, independent of any connection. Sent
// as the first frame of every turn stream (`event: derpr-run`), and returned
// by /active_turn and in a 409 `turn_running` refusal.
export interface TurnRunInfo {
  run_id: string
  kind: 'chat' | 'confirm'
  continue_id: number | null
  done?: boolean
  replay_from?: number
}

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

// ---- POST /api/v1/session/{persona}/assemble (S5 parity inspector; JSON body, DP-500) ----
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

// ---- DP-292 memory import panel ----
export interface MemoryBank {
  bank_id: string
  name?: string | null
  fact_count?: number
  last_document_at?: string | null
}
export interface MemoryDocument {
  // ListDocumentsResponse.items are open objects upstream; these are the
  // fields the panel reads. document_id is the delete/inspect key.
  document_id?: string
  id?: string
  original_text?: string
  content_hash?: string | null
  created_at?: string
  updated_at?: string
  memory_unit_count?: number
  tags?: string[]
  [k: string]: unknown
}
export interface MemoryDocumentList {
  items: MemoryDocument[]
  total?: number
  limit?: number
  offset?: number
}
export interface MemoryOperation {
  id: string
  task_type?: string
  status: string
  document_id?: string | null
  items_count?: number
  created_at?: string
  error_message?: string | null
  retry_count?: number | null
}
// DP-506 purge-and-rebuild job (GET/POST /api/v1/memory/rebuild).
export interface MemoryRebuildPersona {
  name: string
  state: 'waiting' | 'running' | 'done' | 'error' | 'skipped'
  counts: { blocks?: number; queued?: number; adopted?: number; pending?: number; total?: number }
  error: string | null
}
export interface MemoryRebuildState {
  status: 'idle' | 'running' | 'finished'
  started_at: string | null
  finished_at: string | null
  personas: MemoryRebuildPersona[]
  targets?: string[]
}
export interface MemoryOperationList {
  bank_id: string
  total?: number
  operations: MemoryOperation[]
}
export interface UploadResult {
  bank: string
  results: { file: string; status: string; reason?: string; document_id?: string }[]
}

/* KoboldCPP `GET /api/extra/perf`, forwarded verbatim by the adapter (DP-311).
 * Field semantics were verified against a live KCPP 1.115 rather than guessed:
 *   last_input_count  — prompt tokens ingested by the LAST completed gen
 *   last_token_count  — tokens generated by that gen
 *   last_process_time — prefill wall time (s); last_process = ms/token
 *   last_eval_time    — decode wall time (s);  last_eval    = ms/token
 *   idle              — 1 idle, 0 busy (a gen is in flight, from ANY client)
 *   idletime          — seconds since the last state change: while busy this is
 *                       elapsed-in-generation, while idle it is the AGE of the
 *                       last completed gen (which is how we timestamp it)
 *   stop_reason       — -1 none yet, 0 out of tokens, 1 EOS, 2 stop sequence,
 *                       3 aborted (0/1/2 confirmed live; 3 from KCPP source)
 *   total_gens        — COMPLETED generations; the honest "is there a last gen"
 * NOTE: every `last_*` field is frozen for the duration of a generation and
 * only updates on completion — there is no live prefill x/y over the API.
 * NOTE: `stop_reason` is worse than frozen — it RESETS TO 0 when a generation
 * starts, so mid-run it reports "out of tokens" for a run that has not stopped.
 * Consumers must suppress it while `idle === 0`. */
/* DP-311 — `GET /api/extra/prefill`, the engine's projection of the optional
 * kcpp-progress sidecar. This is the ONLY source of live ingestion progress:
 * KoboldCPP prints per-batch prefill counts on stdout and exposes them in no
 * API. `available: false` is a normal state (sidecar not deployed) — consumers
 * must fall back to the last-completed counters in `KoboldPerf`, not error.
 * `run` increments per prefill, so a counter that jumps backwards can be told
 * apart from a new run. */
export interface KoboldPrefill {
  available: boolean
  reason?: string
  phase?: 'idle' | 'prefill' | 'generate'
  processed?: number
  total?: number
  generated?: number
  generate_total?: number
  run?: number
  tok_s?: number | null
}

export interface KoboldPerf {
  last_process: number
  last_eval: number
  last_token_count: number
  last_input_count: number
  last_process_time: number
  last_eval_time: number
  last_process_speed: number
  last_eval_speed: number
  last_seed: number
  total_gens: number
  stop_reason: number
  queue: number
  idle: number
  uptime: number
  idletime: number
}

/** DP-499 — GET /api/v1/inference/servers: the inference router's boxes, for
 *  one statusline row each. `perf`/`prefill` are that box's counters in the
 *  shapes above, from the router's own 3 s poll (null where the box reports
 *  none). `available: false` when derpr is not talking to a router (the
 *  backend switch pinned one box): render the single-server line instead. */
export interface InferenceJob {
  /** persona name, `codr`, `hindsight`, or the caller's IP */
  requester: string
  kind: 'chat' | 'background'
  elapsed_s: number
}

export interface InferenceServer {
  name: string
  primary: boolean
  healthy: boolean
  /** `down`: cannot answer. `unloaded`: answers, but its model is not loaded (lazy) */
  state: 'up' | 'unloaded' | 'down'
  /** seconds between the router's sample of this box and the router answering; null before one */
  sample_age_s: number | null
  model: string | null
  /** busy with work that did not come through the router (direct use of the box) */
  external_busy: boolean
  jobs: InferenceJob[]
  perf: KoboldPerf | null
  prefill: KoboldPrefill | null
}

export interface InferenceServers {
  available: boolean
  reason?: string
  /** chat requests the router is holding for a free box, oldest first */
  held?: InferenceJob[]
  servers?: InferenceServer[]
}

/** DP-438 — GET/PUT/DELETE /api/v1/local_backend: which server `model local`
 *  talks to. `temporary` is lost on restart; `saved` survives it; `env` and
 *  `default` are LOCAL_LLM_URL / the config default. */
export interface LocalBackendState {
  url: string
  label: string | null
  source: 'temporary' | 'saved' | 'env' | 'default'
  presets: { label: string | null; url: string }[]
}

// ---- INFRA panel (DP-439) ---------------------------------------------
// GET /api/v1/infra/status — the proxmox tools' own result dicts, one per
// section; each is `status: 'ok'` with its fields or `status: 'error'` with a
// message, independently.

export interface InfraError {
  status: 'error'
  message?: string
}

export interface InfraGuest {
  vmid: string
  name: string
  kind: 'ct' | 'vm'
  status: string
  lock: string
}

export interface InfraGpu {
  card: string
  vram_total_mib: number
  vram_used_mib: number
  vram_free_mib: number
}

export interface InfraModel {
  name: string
  unit: string
  state: string
  tier: 'hot' | 'cold' | string
  pinned?: boolean
  contextsize?: number
  quantkv?: number
}

export type InfraPve = { status: 'ok'; uptime?: string; guests: InfraGuest[]; inventory_errors?: string[] } | InfraError
export type InfraGpuStatus = { status: 'ok'; host_vmid?: string; gpus: InfraGpu[]; errors?: string[] } | InfraError
export type InfraModels = { status: 'ok'; host_vmid?: string; models: InfraModel[]; note?: string } | InfraError

/** DP-465: one sandbox box's network switches. `null` = could not be read. */
export interface SandboxBox {
  status: 'ok'
  box: 'codr' | 'stagr'
  lan: boolean | null
  wan: boolean | null
  allowlist: string | null
  lan_connections: string[]
  wan_connections: string[]
  log: string[]
  errors?: Record<string, string>
}
export type SandboxNet = { status: 'ok'; boxes: (SandboxBox | InfraError)[] } | InfraError

export interface InfraStatus {
  enabled: boolean
  pve: InfraPve
  gpu: InfraGpuStatus
  models: InfraModels
  net?: SandboxNet
}

export type InfraAction =
  | { action: 'start_guest' | 'stop_guest' | 'reboot_guest'; vmid: string; kind: 'ct' | 'vm' }
  | { action: 'set_active_model'; name: string }
  | { action: 'reboot_node' }
  | { action: 'set_sandbox_net'; box: 'codr' | 'stagr'; zone: 'lan' | 'wan'; state: 'on' | 'off'; confirm?: boolean }

/** The handler's own result dict — `status` plus whatever the tool returned. */
export interface InfraActionResult {
  status: string
  message?: string
  state?: string
  [k: string]: unknown
}
