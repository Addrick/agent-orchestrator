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

// localStorage key for the last-selected persona. Client-side only — the
// engine's PUT /api/v1/model active persona is runtime routing state, not a
// persisted setting, and resets on restart (intentionally not duplicated
// server-side).
const PERSONA_LS_KEY = 'derpr_ui_active_persona'

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
  // Optimistic echo of the just-sent user turn. Rendered as a transient row
  // until an authoritative /transcript re-sync surfaces the persisted one —
  // must be null whenever a re-sync has run (else the row duplicates) and on
  // retry/regen (no new user turn exists).
  userText: string | null
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
  userText: null,
}

export interface DevRow {
  command: string
  response: string
  mutated: boolean
}

// A fired-timer alarm pushed from the engine over GET /voice/alarms (SSE).
// Rendered as a read-only assistant-style line beneath the transcript (it is
// NOT a persisted interaction, so it carries no interaction_id and offers no row
// actions) and accompanied by a short beep.
export interface AlarmLine {
  id: string
  text: string
  channel: string
}

// A short two-tone beep via the Web Audio API — no asset to bundle. Best-effort:
// a browser that blocks autoplay before any user gesture just stays silent.
function playBeep(): void {
  try {
    const Ctx = window.AudioContext || (window as unknown as {
      webkitAudioContext: typeof AudioContext
    }).webkitAudioContext
    const ctx = new Ctx()
    const now = ctx.currentTime
    for (const [i, freq] of [880, 660].entries()) {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = 'sine'
      osc.frequency.value = freq
      const t0 = now + i * 0.22
      gain.gain.setValueAtTime(0.0001, t0)
      gain.gain.exponentialRampToValueAtTime(0.25, t0 + 0.02)
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.2)
      osc.connect(gain).connect(ctx.destination)
      osc.start(t0)
      osc.stop(t0 + 0.2)
    }
    setTimeout(() => void ctx.close(), 800)
  } catch {
    /* no audio context — surface the line silently */
  }
}

export function usePortalStore() {
  const [activePersona, setActivePersona] = useState<string>('assistant')
  const [personaList, setPersonaList] = useState<string[]>([])
  // The LLM model catalog (same source as the `what models` dev command);
  // drives the inspector's model_name dropdown. Persona-independent, loaded once.
  const [modelList, setModelList] = useState<string[]>([])
  // Instruct templates the local renderer understands (engine CHAT_TEMPLATES
  // keys); drives the inspector's chat_template dropdown. Fetched, not
  // hardcoded, so it never drifts from the engine (DP-140). Loaded once.
  const [chatTemplates, setChatTemplates] = useState<string[]>([])
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
  const [alarms, setAlarms] = useState<AlarmLine[]>([])
  // The /assemble dry-run payload for the Raw-req inspector (S5). Lazily
  // fetched by the Raw tab, keyed on the active persona + an optional preview
  // message; null until first fetched.
  const [assembled, setAssembled] = useState<AssembledRequest | null>(null)

  const abortRef = useRef<{ abort: () => void } | null>(null)
  const idFrameRef = useRef<DerprIdFrame | null>(null)
  // True while an SSE stream is in flight. State (`stream.active`) is stale
  // inside the action callbacks' closures, so concurrency guards read this ref
  // instead — without it a regen/confirm racing an active stream interleaves
  // tokens into the shared stream state and leaks the prior abort handle.
  const streamingRef = useRef(false)
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
  // Mirror of activePersona for stream callbacks: lets onDone detect that the
  // user switched persona mid-stream and drop the stale re-sync instead of
  // overwriting the new persona's transcript.
  const activePersonaRef = useRef(activePersona)
  useEffect(() => {
    activePersonaRef.current = activePersona
  }, [activePersona])
  // Mirrors for the debounced LTM-preview callback (DP-257): the async fetch
  // fired off a keystroke must read the current persona/toggle, not the values
  // closed over when the debounce was armed.
  const personaRef = useRef(persona)
  const ltmOnRef = useRef(ltmOn)
  useEffect(() => {
    personaRef.current = persona
    ltmOnRef.current = ltmOn
  })

  // ---- fired-timer alarms (SSE back-channel) ------------------------
  // A timer set from the portal fires back here: the engine pushes the alarm
  // over GET /voice/alarms and we surface it as a chat line + beep. One stream
  // for the session (not per-channel); the alarm carries the channel it was set
  // in. No-op if the voice web routes aren't mounted (EventSource just retries).
  useEffect(() => {
    let es: EventSource | null = null
    try {
      es = new EventSource('/voice/alarms')
    } catch {
      return
    }
    es.onmessage = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as { text?: string; channel?: string }
        if (!data.text) return
        setAlarms((prev) => [
          ...prev,
          {
            id: `alarm-${Date.now()}-${prev.length}`,
            text: data.text as string,
            channel: data.channel || '',
          },
        ])
        playBeep()
      } catch {
        /* malformed event — ignore */
      }
    }
    // The engine endpoint may not be mounted (VOICE_WEB_ENABLED off); let the
    // browser's own EventSource backoff handle reconnects silently.
    es.onerror = () => {}
    return () => es?.close()
  }, [])

  const dismissAlarm = useCallback((id: string) => {
    setAlarms((prev) => prev.filter((a) => a.id !== id))
  }, [])

  // Mid-session fetch failures rethrow from the API client (mock fallback is
  // pre-first-live-success only) — keep the last real state and banner it.
  const UNREACHABLE = 'Engine unreachable — showing last loaded state'

  const refreshTranscript = useCallback(
    async (p: string, channel?: string) => {
      try {
        const cs = await api.getTranscript(p, undefined, channel ?? activeChannelRef.current)
        setChunks(cs)
        setOffline(api.usingMock())
      } catch {
        setBanner(UNREACHABLE)
      }
    },
    [],
  )

  const refreshPersona = useCallback(async (p: string) => {
    try {
      const persObj = await api.getPersona(p)
      setPersona(persObj)
      setOffline(api.usingMock())
    } catch {
      setBanner(UNREACHABLE)
    }
  }, [])

  const refreshChannels = useCallback(async (p: string) => {
    try {
      setChannels(await api.getChannels(p))
    } catch {
      setBanner(UNREACHABLE)
    }
  }, [])

  const loadAll = useCallback(
    async (p: string) => {
      setLoading(true)
      try {
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
      } catch {
        setBanner(UNREACHABLE)
      } finally {
        setLoading(false)
      }
    },
    [refreshTranscript],
  )

  // initial boot
  useEffect(() => {
    ;(async () => {
      try {
      const [serverActive, list, models, templates] = await Promise.all([
        api.getActivePersona(),
        api.listPersonas(),
        api.getModelList(),
        api.getChatTemplates(),
      ])
      // Persona selection is a client preference: the engine's in-memory
      // active persona resets on restart, so localStorage is the boot
      // authority. Fall back to the server's pick when nothing is saved or
      // the saved persona no longer exists; push the saved pick back to the
      // engine so the kobold-native passthrough routes agree with the UI.
      const saved = localStorage.getItem(PERSONA_LS_KEY)
      let active = saved && list.includes(saved) ? saved : serverActive
      if (active !== serverActive) {
        try {
          await api.setActivePersona(active)
        } catch (e) {
          // Engine refused the saved pick (deleted persona, guard) — fall back
          // to its own active persona rather than desyncing UI from engine.
          console.error(e)
          active = serverActive
        }
      }
      setActivePersona(active)
      setPersonaList(list.length ? list : [active])
      setModelList(models)
      setChatTemplates(templates)
      await loadAll(active)
      } catch {
        // A mixed-success boot (one call lands, a later one fails) rethrows
        // past the mock fallback — surface it rather than white-screening.
        setBanner(UNREACHABLE)
        setLoading(false)
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- LTM toggle re-fetch ------------------------------------------
  const toggleLtm = useCallback(async () => {
    if (!persona) return
    const next = !ltmOn
    const prevBlock = ltmBlock
    setLtmOn(next)
    try {
      if (next) {
        setLtmBlock(await api.getLtmBlock(persona.name, ''))
      } else {
        setLtmBlock(null)
      }
      await api.patchPersona(persona.name, { long_term_memory: next })
    } catch (e) {
      // roll back the optimistic flip so the UI matches the server
      console.error(e)
      setLtmOn(!next)
      setLtmBlock(prevBlock)
      setBanner('Failed to update LTM setting — reverted')
    }
  }, [ltmOn, ltmBlock, persona])

  // ---- LTM preview keyed on the composer draft (DP-257) -------------
  // The panel/token-count otherwise show an empty-query block, which the
  // backend resolves to "most recent history turn" — for a GLOBAL persona that
  // drifts across channels and rarely matches what the user is about to send.
  // As the user types we debounce-fetch the block for the real draft so the
  // preview mirrors the per-message recall the engine recomputes at submit.
  // Preview-only: the submit path still sends derpr_user_text untouched.
  const ltmPreviewTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const ltmPreviewSeq = useRef(0)
  const previewLtm = useCallback((draft: string) => {
    if (ltmPreviewTimer.current) clearTimeout(ltmPreviewTimer.current)
    const trimmed = draft.trim()
    // Blank draft → leave the existing empty-query preview in place; don't fire.
    if (!trimmed) return
    ltmPreviewTimer.current = setTimeout(() => {
      const p = personaRef.current
      if (!p || !ltmOnRef.current) return
      const seq = ++ltmPreviewSeq.current
      const pName = p.name
      const chan = activeChannelRef.current
      void (async () => {
        try {
          const blk = await api.getLtmBlock(pName, trimmed, chan)
          // Drop a response superseded by a newer keystroke, an LTM-off toggle,
          // or a persona/channel switch that landed while this was in flight.
          if (
            seq !== ltmPreviewSeq.current ||
            !ltmOnRef.current ||
            personaRef.current?.name !== pName ||
            activeChannelRef.current !== chan
          )
            return
          setLtmBlock(blk)
        } catch {
          /* preview-only — keep the last block on a transient fetch failure */
        }
      })()
    }, 400)
  }, [])

  // ---- persona switch -----------------------------------------------
  const switchPersona = useCallback(
    async (name: string) => {
      const prev = activePersonaRef.current
      setActivePersona(name)
      localStorage.setItem(PERSONA_LS_KEY, name)
      try {
        await api.setActivePersona(name)
      } catch (e) {
        // Engine refused the switch (unknown persona / outage). Revert — the
        // engine still generates under the OLD persona, so advancing the UI
        // would silently log every turn to the wrong persona.
        console.error(e)
        setActivePersona(prev)
        localStorage.setItem(PERSONA_LS_KEY, prev)
        setBanner(`Persona switch failed: ${e instanceof Error ? e.message : String(e)}`)
        return
      }
      // A persona switch resets to its default channel; channel list reloads
      // inside loadAll for the new persona.
      setActiveChannel('web_ui')
      activeChannelRef.current = 'web_ui'
      await loadAll(name)
    },
    [loadAll],
  )

  // ---- create a new persona (DP-231) --------------------------------
  // POST the create body, refresh the picker list, then switch to the new
  // persona so its (real) Inspector is immediately available for the rest of
  // the config (tools, samplers). Returns {ok} so the modal can keep itself
  // open and surface the engine's error (duplicate / invalid name) on failure.
  const createPersona = useCallback(
    async (
      body: Record<string, unknown>,
    ): Promise<{ ok: boolean; name?: string; error?: string }> => {
      try {
        const res = await api.createPersona(body)
        const name = res.persona.name
        const list = await api.listPersonas()
        setPersonaList(list.length ? list : [name])
        await switchPersona(name)
        return { ok: true, name }
      } catch (e) {
        return { ok: false, error: e instanceof Error ? e.message : String(e) }
      }
    },
    [switchPersona],
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
        try {
          if (ltmOn) setLtmBlock(await api.getLtmBlock(persona.name, '', channel))
        } catch {
          setBanner(UNREACHABLE)
        }
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
    (personaName: string, channel: string) => ({
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
      onError: (msg: string) => {
        streamingRef.current = false
        abortRef.current = null
        // A failed CONFIRM resume must re-arm the approve/deny bar — onDone
        // never fires on this path, and leaving resolvingConfirm set keeps the
        // bar permanently disabled ("resolving…") until a page reload.
        setResolvingConfirm(false)
        setStream((s) => ({ ...s, active: false, errored: true, errorMsg: msg }))
      },
      onDone: async () => {
        streamingRef.current = false
        // Authoritative re-sync: the id-frame is an optimization only. The
        // re-fetch surfaces a newly-persisted row, or the trailing ephemeral
        // chunk when the turn (or a chained confirm) parked again — but ONLY
        // if the view still matches what the turn was SENT under (captured at
        // stream start). If the user switched persona or channel mid-stream,
        // the result is stale: writing it would overwrite the transcript and
        // channel rail the new view just loaded, so drop it.
        const stale =
          personaName !== activePersonaRef.current ||
          channel !== activeChannelRef.current
        if (!stale) {
          await refreshTranscript(personaName, channel)
          // A first turn on a brand-new channel materializes it in the DB;
          // reload the channel list so it appears in the rail.
          await refreshChannels(personaName)
        }
        setStream((s) => {
          // A stale turn's row belongs to the old view — clear everything.
          if (stale) return { ...EMPTY_STREAM }
          // The re-fetch above already surfaced the persisted user row, so the
          // optimistic echo must clear even on the kept-visible error/abort
          // treatments — otherwise the turn renders twice.
          if (s.errored) return { ...s, active: false, userText: null }
          if (s.aborted) return { ...s, active: false, userText: null }
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
      // One stream at a time: a second send/regen while one is in flight would
      // interleave both streams' frames into the shared stream state and leak
      // the first abort handle.
      if (streamingRef.current) return
      const trimmed = text.trim()
      if (!trimmed && !retry) return

      // dev command path
      if (trimmed.startsWith('/')) {
        try {
          const resp = await api.devCommand(persona.name, trimmed)
          setDevRow({
            command: trimmed,
            response: resp.response,
            mutated: Boolean(resp.mutated),
          })
          if (resp.mutated) await refreshPersona(persona.name)
          await refreshTranscript(persona.name)
        } catch (e) {
          // Engine down: without this the command silently does nothing
          // (unhandled rejection, no UI feedback).
          console.error(e)
          setBanner('Dev command failed — engine unreachable')
        }
        return
      }

      setBanner(null)
      idFrameRef.current = null
      streamingRef.current = true
      // Echo the user turn immediately; a retry replays the last persisted
      // user turn, so it gets no new echo.
      setStream({ ...EMPTY_STREAM, active: true, userText: retry ? null : trimmed })

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
        buildHandlers(persona.name, activeChannel),
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
      // Same single-stream guard as sendTurn — approve/deny while a stream is
      // in flight would interleave into the shared stream state.
      if (streamingRef.current) return
      setBanner(null)
      idFrameRef.current = null
      streamingRef.current = true
      setResolvingConfirm(true)
      setStream({ ...EMPTY_STREAM, active: true })
      const handle = streamConfirm(
        persona.name,
        approved,
        token,
        buildHandlers(persona.name, activeChannelRef.current),
      )
      abortRef.current = handle
    },
    [persona, buildHandlers],
  )

  const abortTurn = useCallback(async () => {
    await api.abort()
    abortRef.current?.abort()
    abortRef.current = null
    streamingRef.current = false
    // The re-fetch below surfaces the persisted user row — drop the echo.
    // active:false is essential — Conversation passes stream.active as the
    // Composer `streaming` prop, so omitting it leaves the composer stuck on
    // the "■ stop" button and the StreamRow cursor blinking forever.
    setStream((s) => ({ ...s, active: false, aborted: true, userText: null }))
    // An aborted CONFIRM resume never reaches onDone — re-arm the bar.
    setResolvingConfirm(false)
    if (persona) await refreshTranscript(persona.name)
  }, [persona, refreshTranscript])

  // Dismiss re-syncs: an errored turn skips the onDone re-fetch, so the user
  // turn (persisted before generation) is only in the DB — surface it.
  const dismissStream = useCallback(() => {
    streamingRef.current = false
    setResolvingConfirm(false)
    setStream(EMPTY_STREAM)
    if (persona) void refreshTranscript(persona.name)
  }, [persona, refreshTranscript])
  const dismissDevRow = useCallback(() => setDevRow(null), [])

  // ---- regenerate ----------------------------------------------------
  const regen = useCallback(async () => {
    // Retry archives the prior assistant row + makes a new version.
    await sendTurn(' ', true) // non-empty placeholder; engine uses last user turn
  }, [sendTurn])

  // ---- row mutations -------------------------------------------------
  const editRow = useCallback(
    async (id: number, content: string) => {
      // Callers are fire-and-forget (the row editor has already closed), so a
      // rejection here would otherwise vanish — the user believes the edit
      // persisted while the DB still holds the old content.
      try {
        await api.patchInteraction(id, content)
      } catch (e) {
        console.error(e)
        setBanner('Edit failed — row unchanged')
        return
      }
      if (persona) await refreshTranscript(persona.name)
    },
    [persona, refreshTranscript],
  )

  const deleteRow = useCallback(
    async (id: number) => {
      try {
        await api.deleteInteraction(id)
      } catch (e) {
        console.error(e)
        setBanner('Delete failed — row unchanged')
        return
      }
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
        try {
          result = await api.patchPersona(persona.name, patchBody)
        } catch (e) {
          console.error(e)
          setBanner('Persona save failed')
          return { result: 'error', rejected_fields: [], unknown_fields: [] }
        }
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
      try {
        const a = await api.getAssembled(persona.name, message)
        setAssembled(a)
        setOffline(api.usingMock())
      } catch {
        setBanner(UNREACHABLE)
      }
    },
    [persona],
  )

  return {
    // state
    activePersona,
    personaList,
    modelList,
    chatTemplates,
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
    alarms,
    assembled,
    // actions
    dismissAlarm,
    fetchAssembled,
    setViewMode,
    toggleLtm,
    previewLtm,
    switchPersona,
    createPersona,
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
