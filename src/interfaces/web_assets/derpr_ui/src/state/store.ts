/* ============================================================
   Portal store — single source of UI state, hydrated from the engine.
   Chunks keyed by interaction_id / ephemeral_chunk_id (never index).
   The engine is authoritative: after any terminal turn or mutation we
   re-fetch GET /transcript rather than trusting local edits.
   ============================================================ */
import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from '../api/client'
import { streamChat } from '../api/stream'
import type {
  Persona,
  ToolDef,
  ChannelGroup,
  Chunk,
  ToolContext,
  DerprIdFrame,
  PatchPersonaResult,
  DevCommandResponse,
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
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [ltmBlock, setLtmBlock] = useState<string | null>(null)
  const [ltmOn, setLtmOn] = useState<boolean>(true)
  const [viewMode, setViewMode] = useState<ViewMode>('rendered')
  const [offline, setOffline] = useState<boolean>(false)
  const [loading, setLoading] = useState<boolean>(true)
  const [stream, setStream] = useState<StreamState>(EMPTY_STREAM)
  const [devRow, setDevRow] = useState<DevRow | null>(null)
  const [banner, setBanner] = useState<string | null>(null)

  const abortRef = useRef<{ abort: () => void } | null>(null)
  const idFrameRef = useRef<DerprIdFrame | null>(null)

  // ---- loaders -------------------------------------------------------
  const refreshTranscript = useCallback(
    async (p: string) => {
      const cs = await api.getTranscript(p)
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
      if (ltmOn) {
        const blk = await api.getLtmBlock(p, '')
        setLtmBlock(blk)
      }
      setOffline(api.usingMock())
      setLoading(false)
    },
    [ltmOn, refreshTranscript],
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
    }
  }, [ltmOn, persona])

  // ---- persona switch -----------------------------------------------
  const switchPersona = useCallback(
    async (name: string) => {
      setActivePersona(name)
      await api.setActivePersona(name)
      await loadAll(name)
    },
    [loadAll],
  )

  // ---- send a chat turn ---------------------------------------------
  const sendTurn = useCallback(
    async (text: string, retry = false) => {
      if (!persona) return
      const trimmed = text.trim()
      if (!trimmed) return

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
          derpr_user_text: trimmed,
          derpr_retry: retry,
          model: persona.name,
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
        {
          onToken: (delta) =>
            setStream((s) => ({ ...s, text: s.text + delta })),
          onToolStart: (f) =>
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
          onToolResult: (f) =>
            setStream((s) => ({
              ...s,
              tools: s.tools.map((t) =>
                t.call_id === f.call_id
                  ? { ...t, result: f.result, error: f.error }
                  : t,
              ),
            })),
          onIdFrame: (f) => {
            idFrameRef.current = f
            setStream((s) => ({ ...s, responseType: f.response_type }))
          },
          onError: (msg) =>
            setStream((s) => ({
              ...s,
              errored: true,
              errorMsg: msg,
            })),
          onDone: async () => {
            // Authoritative re-sync: the id-frame is an optimization only.
            const frame = idFrameRef.current
            await refreshTranscript(persona.name)
            setStream((s) => {
              // keep an error/aborted transient visible (no committed row)
              if (s.errored) return { ...s, active: false }
              if (s.aborted) return { ...s, active: false }
              // parked? leave the ephemeral chunk to the transcript re-sync
              if (frame?.ephemeral_chunk_id) return { ...EMPTY_STREAM }
              return { ...EMPTY_STREAM }
            })
            abortRef.current = null
          },
        },
      )
      abortRef.current = handle
    },
    [persona, refreshTranscript, refreshPersona],
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

  return {
    // state
    activePersona,
    personaList,
    modelList,
    persona,
    tools,
    channels,
    chunks,
    ltmBlock,
    ltmOn,
    viewMode,
    offline,
    loading,
    stream,
    devRow,
    banner,
    // actions
    setViewMode,
    toggleLtm,
    switchPersona,
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
