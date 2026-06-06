/* ============================================================
   Portal store — single source of UI state, hydrated from the engine.
   Chunks keyed by interaction_id / ephemeral_chunk_id (never index).
   The engine is authoritative: after any terminal turn or mutation we
   re-fetch GET /transcript rather than trusting local edits.
   ============================================================ */
import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from '../api/client'
import { streamChat, streamConfirm } from '../api/stream'
import type {
  Persona,
  ToolDef,
  ChannelGroup,
  Chunk,
  ToolContext,
  DerprIdFrame,
  ToolStartFrame,
  ToolResultFrame,
  PatchPersonaResult,
  DevCommandResponse,
  AssembledRequest,
} from '../types/contracts'

export type ViewMode = 'rendered' | 'context'

// A transient row representing the in-flight assistant turn (not yet persisted).
export interface StreamState {
  active: boolean
  text: string
  reasoning: string | null
  tools: ToolContext[]
  aborted: boolean
  errored: boolean
  errorMsg: string | null
  responseType: string | null
}

const EMPTY_STREAM: StreamState = {
  active: false,
  text: '',
  reasoning: null,
  tools: [],
  aborted: false,
  errored: false,
  errorMsg: null,
  responseType: null,
}

export interface DevRow {
  command: string
  response: string
  mutated: boolean
}

export function usePortalStore() {
  const [activePersona, setActivePersona] = useState<string>('assistant')
  const [personaList, setPersonaList] = useState<string[]>([])
  // The LLM model catalog (same source as the `what models` dev command);
  // drives the inspector's model_name dropdown. Persona-independent, loaded once.
  const [modelList, setModelList] = useState<string[]>([])
  const [persona, setPersona] = useState<Persona | null>(null)
  const [tools, setTools] = useState<ToolDef[]>([])
  const [channels, setChannels] = useState<ChannelGroup[]>([])
  // The channel string the transcript + next submit are scoped to (DP-136 6b).
  // 'web_ui' is the default; switching channels re-fetches the transcript with
  // ?channel=, "+ new channel" sets a fresh tag that persists on first submit.
  const [activeChannel, setActiveChannel] = useState<string>('web_ui')
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [ltmBlock, setLtmBlock] = useState<string | null>(null)
  const [ltmOn, setLtmOn] = useState<boolean>(true)
  const [viewMode, setViewMode] = useState<ViewMode>('rendered')
  const [offline, setOffline] = useState<boolean>(false)
  const [loading, setLoading] = useState<boolean>(true)
  const [stream, setStream] = useState<StreamState>(EMPTY_STREAM)
  const [devRow, setDevRow] = useState<DevRow | null>(null)
  const [banner, setBanner] = useState<string | null>(null)
  // The /assemble dry-run payload for the Raw-req inspector (S5). Lazily
  // fetched by the Raw tab, keyed on the active persona + an optional preview
  // message; null until first fetched.
  const [assembled, setAssembled] = useState<AssembledRequest | null>(null)

  const abortRef = useRef<{ abort: () => void } | null>(null)
  const idFrameRef = useRef<DerprIdFrame | null>(null)
  // True while a CONFIRM-mode write is parked awaiting approval — drives the
  // approve/deny bar's busy state on the ephemeral chunk.
  const [resolvingConfirm, setResolvingConfirm] = useState<boolean>(false)

  // ---- loaders -------------------------------------------------------
  // Transcript is scoped to the active channel (DP-136 6b). The engine honors
  // the persona's memory_mode: a CHANNEL-mode persona isolates per channel; a
  // GLOBAL-mode persona merges all channels regardless of this tag.
  const activeChannelRef = useRef(activeChannel)
  useEffect(() => {
    activeChannelRef.current = activeChannel
  }, [activeChannel])
  const refreshTranscript = useCallback(
    async (p: string) => {
      const cs = await api.getTranscript(p, undefined, activeChannelRef.current)
      setChunks(cs)
      setOffline(api.usingMock())
    },
    [],
  )

  const refreshPersona = useCallback(async (p: string) => {
    const persObj = await api.getPersona(p)
    setPersona(persObj)
    setOffline(api.usingMock())
  }, [])

  const refreshChannels = useCallback(async (p: string) => {
    setChannels(await api.getChannels(p))
  }, [])

  const loadAll = useCallback(
    async (p: string) => {
      setLoading(true)
      const [persObj, toolList, chanList] = await Promise.all([
        api.getPersona(p),
        api.getTools(),
        api.getChannels(p),
      ])
      setPersona(persObj)
      setTools(toolList)
      setChannels(chanList)
      await refreshTranscript(p)
      
      const isLtmOn = persObj.long_term_memory ?? true
      setLtmOn(isLtmOn)
      
      if (isLtmOn) {
        const blk = await api.getLtmBlock(p, '')
        setLtmBlock(blk)
      } else {
        setLtmBlock(null)
      }
      setOffline(api.usingMock())
      setLoading(false)
    },
    [refreshTranscript],
  )

  // initial boot
  useEffect(() => {
    ;(async () => {
      const [active, list, models] = await Promise.all([
        api.getActivePersona(),
        api.listPersonas(),
        api.getModelList(),
      ])
      setActivePersona(active)
      setPersonaList(list.length ? list : [active])
      setModelList(models)
      await loadAll(active)
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- LTM toggle re-fetch ------------------------------------------
  const toggleLtm = useCallback(async () => {
    const next = !ltmOn
    setLtmOn(next)
    if (next && persona) {
      const blk = await api.getLtmBlock(persona.name, '')
      setLtmBlock(blk)
    } else {
      setLtmBlock(null)
    }
    if (persona) {
      api.patchPersona(persona.name, { long_term_memory: next }).catch(e => console.error(e))
    }
  }, [ltmOn, persona])

  // ---- persona switch -----------------------------------------------
  const switchPersona = useCallback(
    async (name: string) => {
      setActivePersona(name)
      await api.setActivePersona(name)
      // A persona switch resets to its default channel; channel list reloads
      // inside loadAll for the new persona.
      setActiveChannel('web_ui')
      activeChannelRef.current = 'web_ui'
      await loadAll(name)
    },
    [loadAll],
  )

  // ---- channel switch / create (6b) ---------------------------------
  // Switch the active channel and re-fetch the transcript scoped to it. The
  // engine honors the persona's memory_mode for isolation.
  const switchChannel = useCallback(
    async (channel: string) => {
      setActiveChannel(channel)
      activeChannelRef.current = channel
      if (persona) {
        await refreshTranscript(persona.name)
        if (ltmOn) setLtmBlock(await api.getLtmBlock(persona.name, '', channel))
      }
    },
    [persona, ltmOn, refreshTranscript],
  )

  // "+ new channel" — just point the active channel at a fresh tag. No row
  // exists yet, so the transcript comes back empty; the channel materializes
  // in the DB (and the channel list) on the first submit.
  const newChannel = useCallback(
    async (rawName?: string) => {
      const base = (rawName || '').trim() || `web_ui_${Date.now().toString(36)}`
      const tag = base.startsWith('web_ui') ? base : `web_ui_${base}`
      await switchChannel(tag)
    },
    [switchChannel],
  )

  // Shared SSE handler set for any turn-producing stream (`/v1/chat/completions`
  // and the `/confirm` resume — same wire protocol). `onConfirm` is a no-op on
  // the wire: a parked write surfaces as the trailing ephemeral chunk after the
  // authoritative `/transcript` re-sync (which the engine also re-emits), so we
  // don't render the confirm frame directly — we just let the re-sync show it.
  // This means a CHAINED confirm (approve → another parked write) re-appears as
  // a new ephemeral chunk with a fresh token, and the approve/deny bar re-arms.
  const buildHandlers = useCallback(
    (personaName: string) => ({
      onToken: (delta: string) =>
        setStream((s) => ({ ...s, text: s.text + delta })),
      onToolStart: (f: ToolStartFrame) =>
        setStream((s) => ({
          ...s,
          tools: [
            ...s.tools,
            {
              call_id: f.call_id,
              group_id: f.group_id,
              tool_name: f.tool_name,
              arguments: f.arguments,
              result: null,
              error: null,
            },
          ],
        })),
      onToolResult: (f: ToolResultFrame) =>
        setStream((s) => ({
          ...s,
          tools: s.tools.map((t) =>
            t.call_id === f.call_id
              ? { ...t, result: f.result, error: f.error }
              : t,
          ),
        })),
      onIdFrame: (f: DerprIdFrame) => {
        idFrameRef.current = f
        setStream((s) => ({ ...s, responseType: f.response_type }))
      },
      onError: (msg: string) =>
        setStream((s) => ({ ...s, errored: true, errorMsg: msg })),
      onDone: async () => {
        // Authoritative re-sync: the id-frame is an optimization only. The
        // re-fetch surfaces a newly-persisted row, or the trailing ephemeral
        // chunk when the turn (or a chained confirm) parked again.
        await refreshTranscript(personaName)
        // A first turn on a brand-new channel materializes it in the DB; reload
        // the channel list so it appears in the rail.
        await refreshChannels(personaName)
        setStream((s) => {
          if (s.errored) return { ...s, active: false }
          if (s.aborted) return { ...s, active: false }
          return { ...EMPTY_STREAM }
        })
        setResolvingConfirm(false)
        abortRef.current = null
      },
    }),
    [refreshTranscript, refreshChannels],
  )

  // ---- send a chat turn ---------------------------------------------
  const sendTurn = useCallback(
    async (text: string, retry = false) => {
      if (!persona) return
      const trimmed = text.trim()
      if (!trimmed && !retry) return

      // dev command path
      if (trimmed.startsWith('/')) {
        const resp = await api.devCommand(persona.name, trimmed)
        setDevRow({
          command: trimmed,
          response: resp.response,
          mutated: Boolean(resp.mutated),
        })
        if (resp.mutated) await refreshPersona(persona.name)
        await refreshTranscript(persona.name)
        return
      }

      setBanner(null)
      idFrameRef.current = null
      setStream({ ...EMPTY_STREAM, active: true })

      const handle = streamChat(
        {
          derpr_user_text: retry && !trimmed ? text : trimmed,
          derpr_retry: retry,
          model: persona.name,
          channel: activeChannel,
          // null = persona leaves it unset; omit so the engine uses its own
          // default rather than forwarding a literal null.
          temperature: persona.temperature ?? undefined,
          top_p: persona.top_p ?? undefined,
          top_k: persona.top_k ?? undefined,
          max_tokens: persona.max_tokens,
          rep_pen: persona.kobold_extras.rep_pen,
          min_p: persona.kobold_extras.min_p,
          tfs: persona.kobold_extras.tfs,
        },
        buildHandlers(persona.name),
      )
      abortRef.current = handle
    },
    [persona, activeChannel, refreshTranscript, refreshPersona, buildHandlers],
  )

  // ---- resolve a parked CONFIRM write (6a) --------------------------
  // Approve/deny a parked write via the dedicated POST /confirm endpoint
  // (NOT a free-text chat turn). `token` is the ephemeral chunk's
  // ephemeral_chunk_id (== the engine's resume token). The continuation
  // streams back over the same SSE protocol; on [DONE] we re-sync from
  // /transcript so the ephemeral chunk becomes a persisted row (approve) or
  // vanishes (deny). A chained write re-parks as a fresh ephemeral chunk.
  const resolveConfirm = useCallback(
    async (token: string, approved: boolean) => {
      if (!persona) return
      setBanner(null)
      idFrameRef.current = null
      setResolvingConfirm(true)
      setStream({ ...EMPTY_STREAM, active: true })
      const handle = streamConfirm(
        persona.name,
        approved,
        token,
        buildHandlers(persona.name),
      )
      abortRef.current = handle
    },
    [persona, buildHandlers],
  )

  const abortTurn = useCallback(async () => {
    await api.abort()
    abortRef.current?.abort()
    abortRef.current = null
    setStream((s) => ({ ...s, aborted: true }))
    if (persona) await refreshTranscript(persona.name)
  }, [persona, refreshTranscript])

  const dismissStream = useCallback(() => setStream(EMPTY_STREAM), [])
  const dismissDevRow = useCallback(() => setDevRow(null), [])

  // ---- regenerate ----------------------------------------------------
  const regen = useCallback(async () => {
    // Retry archives the prior assistant row + makes a new version.
    await sendTurn(' ', true) // non-empty placeholder; engine uses last user turn
  }, [sendTurn])

  // ---- row mutations -------------------------------------------------
  const editRow = useCallback(
    async (id: number, content: string) => {
      await api.patchInteraction(id, content)
      if (persona) await refreshTranscript(persona.name)
    },
    [persona, refreshTranscript],
  )

  const deleteRow = useCallback(
    async (id: number) => {
      await api.deleteInteraction(id)
      if (persona) await refreshTranscript(persona.name)
    },
    [persona, refreshTranscript],
  )

  // ---- persona inspector edits (S4) ---------------------------------
  // Two mutation channels: PATCH /persona for the fields the adapter accepts
  // (prompt, model_name, samplers, memory_mode, …) and dev_command for the ones
  // it doesn't (thinking_level, tools, tool_policy). savePersona runs the PATCH,
  // then any dev_commands, then a single authoritative refetch. Rejections from
  // the PATCH are returned so the inspector can surface them.
  const savePersona = useCallback(
    async (
      patchBody: Record<string, unknown>,
      devCommands: string[] = [],
    ): Promise<PatchPersonaResult> => {
      let result: PatchPersonaResult = {
        result: 'success',
        rejected_fields: [],
        unknown_fields: [],
      }
      if (!persona) return result
      if (Object.keys(patchBody).length) {
        result = await api.patchPersona(persona.name, patchBody)
      }
      for (const cmd of devCommands) {
        await api.devCommand(persona.name, cmd)
      }
      await refreshPersona(persona.name)
      return result
    },
    [persona, refreshPersona],
  )

  // Tools tab toggles route through `set tools …` (the DP-120 dev_command path),
  // since enabled_tools is not a PATCH key. Returns the response so the pane can
  // surface an insecure-composition warning. Refreshes persona on mutation.
  const runToolsCommand = useCallback(
    async (command: string): Promise<DevCommandResponse> => {
      if (!persona) return { response: 'no persona', mutated: false }
      const resp = await api.devCommand(persona.name, command)
      if (resp.mutated) await refreshPersona(persona.name)
      return resp
    },
    [persona, refreshPersona],
  )

  // ---- assemble dry-run (S5 Raw-req parity inspector) ---------------
  // Fetch the exact request the engine would send for the active persona,
  // optionally previewing a hypothetical new user turn (`message`). Keyed on
  // the active persona so switching personas re-derives it.
  const fetchAssembled = useCallback(
    async (message = '') => {
      if (!persona) return
      const a = await api.getAssembled(persona.name, message)
      setAssembled(a)
      setOffline(api.usingMock())
    },
    [persona],
  )

  return {
    // state
    activePersona,
    personaList,
    modelList,
    persona,
    tools,
    channels,
    activeChannel,
    chunks,
    ltmBlock,
    ltmOn,
    viewMode,
    offline,
    loading,
    stream,
    resolvingConfirm,
    devRow,
    banner,
    assembled,
    // actions
    fetchAssembled,
    setViewMode,
    toggleLtm,
    switchPersona,
    switchChannel,
    newChannel,
    resolveConfirm,
    sendTurn,
    abortTurn,
    dismissStream,
    dismissDevRow,
    regen,
    editRow,
    deleteRow,
    savePersona,
    runToolsCommand,
    refreshTranscript,
    refreshPersona,
    setPersona,
    setBanner,
  }
}

export type PortalStore = ReturnType<typeof usePortalStore>
