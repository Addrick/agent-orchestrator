/* ============================================================
   API client — the ONLY module that talks to the engine adapter.
   Live-first: every call hits the real endpoint; if the engine is
   unreachable (no backend during build/dev), it falls back to the
   mock fixtures so the UI renders. `usingMock()` reflects the last
   outcome so the chrome can surface an "offline / mock" indicator.

   Single-user, single-channel: user_identifier="portal", channel="web_ui".
   Chunks are addressed by id, never by array index.
   ============================================================ */
import {
  MOCK_PERSONA,
  MOCK_TOOLS,
  MOCK_CHANNELS,
  MOCK_LTM_BLOCK,
  MOCK_TRANSCRIPT,
  MOCK_VERSIONS_1042,
  MOCK_ASSEMBLED,
} from './mock'
import type {
  Persona,
  ToolDef,
  ChannelGroup,
  ChannelItem,
  ChannelRow,
  ChannelSource,
  Chunk,
  TranscriptResponse,
  ToolsCatalog,
  LtmBlockResponse,
  PatchPersonaResult,
  VersionsResponse,
  DevCommandResponse,
  AssembledRequest,
} from '../types/contracts'

// Same-origin in production (served under /derpr by the adapter); the dev
// server proxies /api and /v1 to :5003 (see vite.config.ts).
const BASE = ''

let _usingMock = false
export const usingMock = () => _usingMock

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path, { headers: { Accept: 'application/json' } })
  if (!r.ok) throw new Error(`${path} → ${r.status}`)
  return (await r.json()) as T
}

async function liveOr<T>(live: () => Promise<T>, mock: () => T): Promise<T> {
  try {
    const v = await live()
    _usingMock = false
    return v
  } catch {
    _usingMock = true
    return mock()
  }
}

// ---- persona / model -------------------------------------------------
export function getPersona(name: string): Promise<Persona> {
  return liveOr(
    () => getJSON<Persona>(`/api/v1/persona/${encodeURIComponent(name)}`),
    () => MOCK_PERSONA,
  )
}

// Persona list for the picker. Personas come from /v1/models (OpenAI-style
// data[].id), NOT /api/v1/models/list — the latter returns the LLM model
// catalog (gpt-4o, claude-*, …), which would wrongly fill the picker with
// ~150 model ids. The mock fallback returns persona-ish names.
export function listPersonas(): Promise<string[]> {
  return liveOr(
    async () =>
      (await getJSON<{ data: { id: string }[] }>(`/v1/models`)).data.map((m) => m.id),
    () => [MOCK_PERSONA.name, 'gemini', 'claude'],
  )
}

// The LLM model catalog (for a future model dropdown, S4). Distinct from the
// persona list above.
export function getModelList(): Promise<string[]> {
  return liveOr(
    async () => (await getJSON<{ models: string[] }>(`/api/v1/models/list`)).models,
    () => ['gpt-4o-mini', 'gemini-2.5-flash', 'claude-sonnet-4-5-20250929'],
  )
}

export function getActivePersona(): Promise<string> {
  return liveOr(
    async () => (await getJSON<{ result: string }>(`/api/v1/model`)).result,
    () => MOCK_PERSONA.name,
  )
}

export async function setActivePersona(name: string): Promise<void> {
  try {
    await fetch(`${BASE}/api/v1/model`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: name }),
    })
    _usingMock = false
  } catch {
    _usingMock = true
  }
}

export async function patchPersona(
  name: string,
  body: Record<string, unknown>,
): Promise<PatchPersonaResult> {
  const r = await fetch(`${BASE}/api/v1/persona/${encodeURIComponent(name)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = (await r.json()) as PatchPersonaResult
  return data
}

// ---- tools -----------------------------------------------------------
export function getTools(): Promise<ToolDef[]> {
  return liveOr(
    async () => (await getJSON<ToolsCatalog>(`/api/v1/tools/catalog`)).tools,
    () => MOCK_TOOLS,
  )
}

// ---- channels --------------------------------------------------------
// GET /api/v1/channels (DP-136 6b) returns the distinct channels seen in
// history for the active persona, each with a derived source tag. We group
// them by source for the channel rail. The engine always includes a synthetic
// `web_ui` row so a fresh persona still offers a default channel.
const SOURCE_LABEL: Record<ChannelSource, string> = {
  web: 'Web UI',
  dsc: 'Discord',
  zmd: 'Zammad',
  gml: 'Gmail',
}

export function getChannels(activePersona: string): Promise<ChannelGroup[]> {
  return liveOr(
    async () => {
      const { channels } = await getJSON<{ channels: ChannelRow[] }>(
        `/api/v1/channels?persona=${encodeURIComponent(activePersona)}`,
      )
      const bySource = new Map<ChannelSource, ChannelItem[]>()
      for (const c of channels) {
        const src = (c.source || 'web') as ChannelSource
        const item: ChannelItem = {
          id: `${c.channel}:${c.server_id ?? ''}`,
          channel: c.channel,
          name: c.channel,
          source: src,
          persona: activePersona,
          active: c.channel === 'web_ui',
          preview: c.count ? `${c.count} msgs` : 'no history yet',
        }
        const list = bySource.get(src) ?? []
        list.push(item)
        bySource.set(src, list)
      }
      const order: ChannelSource[] = ['web', 'dsc', 'zmd', 'gml']
      return order
        .filter((s) => bySource.has(s))
        .map((s) => ({ group: SOURCE_LABEL[s], items: bySource.get(s)! }))
    },
    () => MOCK_CHANNELS,
  )
}

// ---- transcript ------------------------------------------------------
// `channel` (DP-136 6b) scopes the transcript; the engine honors the persona's
// memory_mode (CHANNEL isolates, GLOBAL merges). Omitting it keeps the legacy
// global-history view.
export function getTranscript(
  persona: string,
  maxTurns?: number,
  channel?: string,
): Promise<Chunk[]> {
  const qs = new URLSearchParams()
  if (maxTurns) qs.set('max_turns', String(maxTurns))
  if (channel) qs.set('channel', channel)
  const q = qs.toString() ? `?${qs}` : ''
  return liveOr(
    async () =>
      (await getJSON<TranscriptResponse>(
        `/api/v1/session/${encodeURIComponent(persona)}/transcript${q}`,
      )).chunks,
    () => MOCK_TRANSCRIPT.map((c) => ({ ...c })),
  )
}

// ---- ltm -------------------------------------------------------------
export function getLtmBlock(
  persona: string,
  query: string,
  channel?: string,
): Promise<string | null> {
  const qs = new URLSearchParams({ query })
  if (channel) qs.set('channel', channel)
  return liveOr(
    async () =>
      (await getJSON<LtmBlockResponse>(
        `/api/v1/session/${encodeURIComponent(persona)}/ltm_block?${qs}`,
      )).block,
    () => MOCK_LTM_BLOCK,
  )
}

// ---- versions --------------------------------------------------------
export function getVersions(id: number): Promise<VersionsResponse> {
  return liveOr(
    () => getJSON<VersionsResponse>(`/api/v1/interaction/${id}/versions`),
    () => MOCK_VERSIONS_1042,
  )
}

export async function selectVersion(id: number, k: number): Promise<VersionsResponse> {
  const r = await fetch(`${BASE}/api/v1/interaction/${id}/select_version/${k}`, {
    method: 'POST',
  })
  if (!r.ok) throw new Error(`select_version → ${r.status}`)
  return (await r.json()) as VersionsResponse
}

// ---- row mutations ---------------------------------------------------
export async function patchInteraction(id: number, content: string): Promise<void> {
  const r = await fetch(`${BASE}/api/v1/interaction/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  })
  if (!r.ok) throw new Error(`patch interaction → ${r.status}`)
}

export async function deleteInteraction(
  id: number,
): Promise<{ result: string; already_suppressed: boolean }> {
  const r = await fetch(`${BASE}/api/v1/interaction/${id}`, { method: 'DELETE' })
  if (!r.ok) throw new Error(`delete interaction → ${r.status}`)
  return (await r.json()) as { result: string; already_suppressed: boolean }
}

// ---- dev command -----------------------------------------------------
export async function devCommand(
  persona: string,
  command: string,
): Promise<DevCommandResponse> {
  const r = await fetch(`${BASE}/api/v1/persona/${encodeURIComponent(persona)}/dev_command`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command }),
  })
  // 400 returns { response: "Not a dev command" } — still JSON.
  return (await r.json()) as DevCommandResponse
}

// ---- assemble (S5 parity inspector) ----------------------------------
// Dry-run of the exact request the engine would send, from the shared live
// builder. On the live path source='engine.dry_run' (parity verified); the
// mock fallback returns source='client_fallback' so the banner flags drift.
export function getAssembled(
  persona: string,
  message = '',
  retry = false,
): Promise<AssembledRequest> {
  const q = new URLSearchParams({ message, retry: String(retry) }).toString()
  return liveOr(
    () =>
      getJSON<AssembledRequest>(
        `/api/v1/session/${encodeURIComponent(persona)}/assemble?${q}`,
      ),
    () => MOCK_ASSEMBLED,
  )
}

// ---- abort -----------------------------------------------------------
export async function abort(): Promise<void> {
  try {
    await fetch(`${BASE}/api/v1/abort`, { method: 'POST' })
  } catch {
    /* best-effort */
  }
}
