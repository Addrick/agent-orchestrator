import type { PortalStore } from '../state/store'
import { splitThink, policyLabel } from '../state/util'
import { MessageRow } from './MessageRow'
import { StreamRow } from './StreamRow'
import { BudgetBar } from './BudgetBar'
import { ContextView } from './ContextView'
import { Composer } from './Composer'

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
    stream,
    devRow,
    loading,
    sendTurn,
    abortTurn,
    dismissStream,
    dismissDevRow,
    regen,
    editRow,
    deleteRow,
    resolveConfirm,
    resolvingConfirm,
    refreshTranscript,
  } = store

  const onResync = () => persona && refreshTranscript(persona.name)

  const historyText = chunks
    .filter((c) => !c.ephemeral)
    .map((c) => splitThink(c.content).body)
    .join('\n')

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

      <div className="scroll">
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

              {(() => {
                const lastUserChunkIndex = chunks.reduce(
                  (acc, c, i) => (c.role === 'user' && !c.ephemeral ? i : acc),
                  -1,
                )
                const lastAssistantChunkIndex = chunks.reduce(
                  (acc, c, i) =>
                    c.role === 'assistant' && !c.ephemeral ? i : acc,
                  -1,
                )
                // LTM author's-note row renders after the FIRST real user turn,
                // located by identity (not positional index===1, which vanished
                // on a single-chunk transcript and mispositioned when chunk[0]
                // was an assistant turn) — DP-132 #9.
                const firstUserChunkIndex = chunks.findIndex(
                  (c) => c.role === 'user' && !c.ephemeral,
                )
                return chunks.map((c, i) => (
                  <RenderedSlot
                    key={c.interaction_id ?? c.ephemeral_chunk_id ?? `slot-${i}`}
                    showLtm={i === firstUserChunkIndex}
                    ltmOn={ltmOn}
                    ltmBlock={ltmBlock}
                    persona={persona}
                  >
                    <MessageRow
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
                      onResync={onResync}
                      onResolveConfirm={resolveConfirm}
                      resolvingConfirm={resolvingConfirm}
                    />
                  </RenderedSlot>
                ))
              })()}

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
                onRegen={regen}
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
        </div>
      </div>

      <Composer
        ltmOn={ltmOn}
        onToggleLtm={toggleLtm}
        onSend={(t) => sendTurn(t)}
        onAbort={abortTurn}
        streaming={stream.active}
      />
    </div>
  )
}

// Injects the LTM row right after the first user turn (matches the prototype's
// author's-note placement). The caller locates that turn by identity and passes
// `showLtm`, so the row no longer depends on a positional index.
function RenderedSlot({
  showLtm: showLtmSlot,
  ltmOn,
  ltmBlock,
  persona,
  children,
}: {
  showLtm: boolean
  ltmOn: boolean
  ltmBlock: string | null
  persona: { name: string }
  children: React.ReactNode
}) {
  const showLtm = ltmOn && ltmBlock && showLtmSlot
  return (
    <>
      {showLtm && (
        <div className="ltmrow">
          <div className="lh">
            <span className="lbl">◈ LTM recalled · injected as author's-note</span>
            <span className="lbl" style={{ marginLeft: 'auto' }}>
              /session/{persona.name}/ltm_block
            </span>
          </div>
          <div className="text">{ltmBlock}</div>
        </div>
      )}
      {children}
    </>
  )
}
