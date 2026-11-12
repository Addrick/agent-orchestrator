import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { PortalStore } from '../state/store'
import { estimateTokens, policyLabel } from '../state/util'
import { MessageRow } from './MessageRow'
import { StreamRow } from './StreamRow'
import { BudgetBar } from './BudgetBar'
import { ContextView } from './ContextView'
import { Composer } from './Composer'
import { LtmDock } from './LtmDock'

export function Conversation({ store }: { store: PortalStore }) {
  const {
    persona,
    chunks,
    tools,
    ltmBlock,
    ltmOn,
    viewMode,
    setViewMode,
    toggleLtm,
    previewLtm,
    stream,
    devRow,
    loading,
    sendTurn,
    abortTurn,
    dismissStream,
    dismissDevRow,
    regen,
    continueTurn,
    restoredDraft,
    editRow,
    deleteRow,
    resolveConfirm,
    resolvingConfirm,
    refreshTranscript,
    alarms,
    dismissAlarm,
  } = store

  // Stable identity so the memoized MessageRow isn't re-rendered per token.
  const personaName = persona?.name
  const onResync = useCallback(() => {
    if (personaName) void refreshTranscript(personaName)
  }, [personaName, refreshTranscript])

  // DP-511: the LTM preview is collapsed by default on every load.
  const [ltmOpen, setLtmOpen] = useState(false)
  const ltmTokens = ltmOn && ltmBlock ? estimateTokens(ltmBlock) : null

  // Auto-scroll: follow new content (chunks, stream frames, dev rows) but only while the user is pinned to the bottom. DP-510: the
  // follow is released by any upward scroll — not by distance from the bottom,
  // which kept it armed through the first frames of an animated wheel notch so
  // the next token yanked the view back — and re-armed only near the bottom.
  // Programmatic follow scrolls only ever move down; a content shrink that
  // clamps scrollTop lands exactly on the bottom.
  const scrollRef = useRef<HTMLDivElement>(null)
  const followRef = useRef(true)
  const lastTopRef = useRef(0)
  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    const gap = el.scrollHeight - el.scrollTop - el.clientHeight
    if (gap < 2) followRef.current = true
    else if (el.scrollTop < lastTopRef.current) followRef.current = false
    else if (gap < 80) followRef.current = true
    lastTopRef.current = el.scrollTop
  }
  const onWheel = (e: React.WheelEvent) => {
    if (e.deltaY < 0) followRef.current = false
  }
  useEffect(() => {
    const el = scrollRef.current
    if (el && followRef.current) {
      el.scrollTop = el.scrollHeight
      lastTopRef.current = el.scrollTop
    }
  }, [chunks, stream, devRow, loading, viewMode, alarms])
  // The scroller shrinks when the LTM dock opens or is dragged taller (and when
  // the draft textarea grows); keep a pinned view on the bottom through it.
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      if (followRef.current) {
        el.scrollTop = el.scrollHeight
        lastTopRef.current = el.scrollTop
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Feeds only the BudgetBar; the filter + join over the whole
  // transcript is too heavy to redo on every per-frame stream re-render.
  const historyText = useMemo(
    () =>
      chunks
        .filter((c) => !c.ephemeral)
        .map((c) => c.content)
        .join('\n'),
    [chunks],
  )

  // Identity-based row positions (DP-132 #9); memoized so streaming re-renders
  // skip the full-transcript scans.
  const { lastUserChunkIndex, lastAssistantChunkIndex } =
    useMemo(() => {
      const lastUser = chunks.reduce(
        (acc, c, i) => (c.role === 'user' && !c.ephemeral ? i : acc),
        -1,
      )
      const lastAssistant = chunks.reduce(
        (acc, c, i) => (c.role === 'assistant' && !c.ephemeral ? i : acc),
        -1,
      )
      return {
        lastUserChunkIndex: lastUser,
        lastAssistantChunkIndex: lastAssistant,
      }
    }, [chunks])

  // DP-447: CONTINUE extends the last row when it is an assistant reply with
  // no tool steps, on a persona that routes to the local server.
  const lastChunk = chunks[chunks.length - 1]
  const route = persona?.effective_model_name ?? persona?.model_name
  const canContinue =
    route === 'local' &&
    lastChunk != null &&
    !lastChunk.ephemeral &&
    lastChunk.interaction_id != null &&
    lastChunk.role === 'assistant' &&
    !(lastChunk.tool_context || []).length
  const onContinue = useCallback(() => void continueTurn(), [continueTurn])
  // The row being continued is drawn by the stream row while it grows.
  const hiddenId = stream.active || stream.aborted ? stream.continuingId : null

  return (
    <div className="col convo">
      <div className="convohead">
        <div className="av">{(persona?.name || 'AS').slice(0, 2).toUpperCase()}</div>
        <span className="title">
          {persona ? `${persona.display_name || persona.name} · ${persona.name}` : '—'}
        </span>
        {persona && (
          <>
            <span className="chip">
              <span className="dot" />
              {persona.model_name}
            </span>
            <span className="chip mem">
              <span className="dot" />
              {persona.memory_mode}
            </span>
            <span className="chip">
              <span className="dot" />
              {policyLabel(persona.tool_policy)}
            </span>
            {persona.security_blocked && (
              <span className="chip blocked">security-blocked</span>
            )}
          </>
        )}
        <span className="grow" />
        <div className="seg">
          <button
            id="seg-rendered"
            className={viewMode === 'rendered' ? 'on' : ''}
            onClick={() => setViewMode('rendered')}
          >
            RENDERED
          </button>
          <button
            id="seg-context"
            className={viewMode === 'context' ? 'on' : ''}
            onClick={() => setViewMode('context')}
          >
            CONTEXT ↦ LLM
          </button>
        </div>
      </div>

      {viewMode === 'rendered' && persona && (
        <BudgetBar
          persona={persona}
          systemPrompt={persona.prompt}
          ltmBlock={ltmBlock}
          ltmOn={ltmOn}
          historyText={historyText}
        />
      )}

      <div className="scroll" ref={scrollRef} onScroll={onScroll} onWheel={onWheel}>
        <div className="transcript" id="transcript">
          {loading && <div className="dimrow">loading transcript…</div>}

          {!loading && viewMode === 'rendered' && persona && (
            <>
              {/* pinned system prompt — from persona, NOT a transcript chunk */}
              <div className="sysrow">
                <div className="lh">
                  <span className="lbl">System · persona prompt</span>
                  <span className="lbl" style={{ marginLeft: 'auto' }}>
                    GET /persona/{persona.name}
                  </span>
                </div>
                <div className="text">{persona.prompt}</div>
              </div>

              {chunks.map((c, i) => c.interaction_id != null && c.interaction_id === hiddenId ? null : (
                  <MessageRow
                    key={c.interaction_id ?? c.ephemeral_chunk_id ?? `slot-${i}`}
                    chunk={c}
                    tools={tools}
                    isLastUser={
                      i === lastUserChunkIndex &&
                      lastUserChunkIndex > lastAssistantChunkIndex
                    }
                    isLastAssistant={
                      i === lastAssistantChunkIndex &&
                      lastAssistantChunkIndex > lastUserChunkIndex
                    }
                    onEdit={editRow}
                    onDelete={deleteRow}
                    onRegen={regen}
                    onContinue={canContinue && i === chunks.length - 1 ? continueTurn : undefined}
                    onResync={onResync}
                    onResolveConfirm={resolveConfirm}
                    resolvingConfirm={resolvingConfirm}
                  />
              ))}

              {/* optimistic echo of the just-sent user turn — replaced by the
                  persisted row on the [DONE] /transcript re-sync. Not a Chunk:
                  it has no interaction_id, so it must never get MessageRow's
                  edit/del/retry affordances or count toward chevron gating. */}
              {stream.userText != null && (
                <div className="msg">
                  <div className="gut">
                    <div className="av user">U</div>
                  </div>
                  <div className="bd">
                    <div className="meta">
                      <span className="who user">portal</span>
                      <span className="ts" />
                      <span className="idtag">sending…</span>
                    </div>
                    <div className="text">{stream.userText}</div>
                  </div>
                </div>
              )}

              <StreamRow
                stream={stream}
                tools={tools}
                onDismiss={dismissStream}
                onRegen={stream.continuingId != null ? onContinue : regen}
              />

              {devRow && (
                <div className="devrow">
                  <div className="lh">
                    <span className="lbl">
                      ⌘ dev-command{devRow.mutated ? ' · mutated persona' : ''}
                    </span>
                    <button className="mini" style={{ marginLeft: 'auto' }} onClick={dismissDevRow}>
                      dismiss
                    </button>
                  </div>
                  <div className="text" style={{ color: 'var(--ink-faint)' }}>
                    {devRow.command}
                  </div>
                  <div className="text">{devRow.response}</div>
                </div>
              )}

            </>
          )}

          {!loading && viewMode === 'context' && persona && (
            <ContextView
              persona={persona}
              chunks={chunks}
              ltmBlock={ltmBlock}
              ltmOn={ltmOn}
            />
          )}

          {/* fired-timer alarms (DP-238): pushed from the engine over SSE, NOT
              transcript chunks — read-only assistant-style lines that beep on
              arrival and dismiss locally. Shown in both view modes so an alarm
              is never missed. */}
          {alarms.map((a) => (
            <div className="msg alarmrow" key={a.id}>
              <div className="gut">
                <div className="av">⏰</div>
              </div>
              <div className="bd">
                <div className="meta">
                  <span className="who">timer</span>
                  {a.channel && <span className="idtag">{a.channel}</span>}
                  <button
                    className="mini"
                    style={{ marginLeft: 'auto' }}
                    onClick={() => dismissAlarm(a.id)}
                  >
                    dismiss
                  </button>
                </div>
                <div className="text">{a.text}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* DP-511: docked outside the scroller — it belongs to the next message,
          not the history (DP-431), and its height changes only when dragged. */}
      {ltmOpen && ltmOn && ltmBlock && persona && <LtmDock block={ltmBlock} personaName={persona.name} />}

      <Composer
        ltmOn={ltmOn}
        onToggleLtm={toggleLtm}
        ltmTokens={ltmTokens}
        ltmOpen={ltmOpen}
        onToggleLtmOpen={() => setLtmOpen((o) => !o)}
        onSend={(t) => sendTurn(t)}
        onContinue={canContinue ? onContinue : undefined}
        restoredDraft={restoredDraft}
        onAbort={abortTurn}
        streaming={stream.active}
        onDraftChange={previewLtm}
      />
    </div>
  )
}
