import { Fragment, memo, useState } from 'react'
import type { Chunk, ToolDef } from '../types/contracts'
import { joinRawTurn, rawThinkState } from '../state/util'
import { MessageBody } from './MessageBody'
import { ReasoningFold } from './ReasoningFold'
import { ToolCard } from './ToolCard'
import { VersionChevrons } from './VersionChevrons'

interface Props {
  chunk: Chunk
  tools: ToolDef[]
  onEdit: (id: number, content: string, rawThink?: boolean) => Promise<boolean> | void
  // DP-447: set on the conversation's last row when it can be continued; the
  // editor then offers "save & continue".
  onContinue?: (thinkOpen?: boolean) => void
  onDelete: (id: number) => void
  onRegen: () => void
  onResync: () => void
  onResolveConfirm: (token: string, approved: boolean) => void
  resolvingConfirm: boolean
  isLastUser?: boolean
  isLastAssistant?: boolean
}

// Memoized: during a stream the transcript's chunk objects and the store's
// action callbacks are referentially stable, so per-token re-renders of the
// tree skip every persisted row.
export const MessageRow = memo(function MessageRow({
  chunk: c,
  tools,
  onEdit,
  onDelete,
  onRegen,
  onResync,
  onResolveConfirm,
  resolvingConfirm,
  isLastUser,
  isLastAssistant,
  onContinue,
}: Props) {
  // Reasoning is its own field (DP-434); content is the reply alone.
  const reasoning = c.role === 'assistant' ? c.reasoning : null
  const body = c.content
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(body)

  const who = c.role === 'assistant' ? 'assistant' : 'portal'
  const av = c.role === 'assistant' ? 'AS' : 'U'

  const idTag = c.ephemeral ? (
    <span className="idtag eph">ephemeral · {c.ephemeral_chunk_id || 'pending'}</span>
  ) : c.interaction_id != null ? (
    <span className="idtag">#{c.interaction_id}</span>
  ) : (
    <span className="idtag">unaddressable</span>
  )

  // DP-447: an assistant turn is edited whole — thinking and reply in one
  // raw text. Not a turn with tool steps: its stored reasoning is every step's
  // thinking joined, which one block would flatten (the engine refuses it).
  const rawEdit = c.role === 'assistant' && !(c.tool_context || []).length
  const beginEdit = () => {
    setDraft(rawEdit ? joinRawTurn(reasoning, body) : body)
    setEditing(true)
  }
  const saveEdit = async (thenContinue = false) => {
    setEditing(false)
    if (c.interaction_id == null) return
    const saved = await onEdit(c.interaction_id, draft, rawEdit)
    if (thenContinue && saved !== false && onContinue) {
      const state = rawThinkState(draft)
      onContinue(state === 'open' ? true : state ? false : undefined)
    }
  }

  return (
    <div className={'msg' + (c.ephemeral ? ' ephemeral' : '')}>
      <div className="gut">
        <div className={'av ' + c.role}>{av}</div>
      </div>
      <div className="bd">
        <div className="meta">
          <span className={'who ' + c.role}>{who}</span>
          <span className="ts" />
          {idTag}
          {c.has_versions && c.interaction_id != null && (
            <VersionChevrons interactionId={c.interaction_id} onResync={onResync} />
          )}
          {c.ephemeral && (
            <span
              className="chip"
              style={{ color: 'var(--write)', borderColor: 'rgba(231,173,98,.4)' }}
            >
              <span className="dot" style={{ background: 'var(--write)' }} />
              awaiting approval
            </span>
          )}
        </div>

        {/* Prompt order (DP-437): each step's thinking, then its prose
            (DP-440), above its calls; the
            chunk's own reasoning is what came after the last tool result. */}
        {(c.tool_context || []).map((tc) => (
          <Fragment key={tc.call_id}>
            {tc.reasoning && <ReasoningFold reasoning={tc.reasoning} />}
            {tc.prose && (
              <div className="text">
                <MessageBody content={tc.prose} />
              </div>
            )}
            <ToolCard tc={tc} tools={tools} />
          </Fragment>
        ))}

        {/* The raw editor below already holds the thinking. */}
        {reasoning && !(editing && rawEdit) && <ReasoningFold reasoning={reasoning} />}

        {editing ? (
          <div style={{ marginTop: 10 }}>
            <textarea
              className="editbox"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              autoFocus
            />
            <div className="editbar">
              <button className="btn approve" onClick={() => void saveEdit()}>
                save
              </button>
              {onContinue && (
                <button
                  className="btn approve"
                  title="save, then let the model carry on from exactly where this text stops"
                  onClick={() => void saveEdit(true)}
                >
                  save &amp; continue
                </button>
              )}
              <button className="btn" onClick={() => setEditing(false)}>
                cancel
              </button>
            </div>
          </div>
        ) : (
          body &&
          body.trim() && (
            <div
              className="text"
              style={{
                marginTop: reasoning || (c.tool_context || []).length ? 10 : 0,
              }}
            >
              <MessageBody content={body} />
            </div>
          )
        )}

        {/* CONFIRM bar for ephemeral parked write — resolves via POST /confirm
            (DP-136 6a) with the ephemeral_chunk_id as the resume token. */}
        {c.ephemeral && (
          <div className="confirm">
            <span className="lbl">CONFIRM</span>
            <button
              className="btn approve"
              disabled={resolvingConfirm || !c.ephemeral_chunk_id}
              title="approve & run the parked write"
              onClick={() =>
                c.ephemeral_chunk_id &&
                onResolveConfirm(c.ephemeral_chunk_id, true)
              }
            >
              ✓ approve &amp; run
            </button>
            <button
              className="btn deny"
              disabled={resolvingConfirm || !c.ephemeral_chunk_id}
              title="deny the parked write"
              onClick={() =>
                c.ephemeral_chunk_id &&
                onResolveConfirm(c.ephemeral_chunk_id, false)
              }
            >
              ✕ deny
            </button>
            <span className="note">
              {resolvingConfirm
                ? 'resolving…'
                : 'resolves via /confirm · tool_policy = CONFIRM'}
            </span>
          </div>
        )}
      </div>

      {/* row hover actions — not on ephemeral */}
      {!c.ephemeral && !editing && c.interaction_id != null && (
        <div className="rowacts">
          {c.role === 'assistant' && isLastAssistant && (
            <button className="ract" title="regenerate" onClick={onRegen}>
              ⟲ regen
            </button>
          )}
          {c.role === 'user' && isLastUser && (
            <button className="ract" title="retry" onClick={onRegen}>
              ⟲ retry
            </button>
          )}
          <button className="ract" title="edit" onClick={beginEdit}>
            ✎ edit
          </button>
          <button
            className="ract danger"
            title="suppress"
            onClick={() => onDelete(c.interaction_id as number)}
          >
            ✕ del
          </button>
        </div>
      )}
    </div>
  )
})
