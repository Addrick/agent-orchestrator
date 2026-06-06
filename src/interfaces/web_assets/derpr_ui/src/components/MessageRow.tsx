import { useState } from 'react'
import type { Chunk, ToolDef } from '../types/contracts'
import { splitThink } from '../state/util'
import { ReasoningFold } from './ReasoningFold'
import { ToolCard } from './ToolCard'
import { VersionChevrons } from './VersionChevrons'

interface Props {
  chunk: Chunk
  tools: ToolDef[]
  onEdit: (id: number, content: string) => void
  onDelete: (id: number) => void
  onRegen: () => void
  onResync: () => void
  onResolveConfirm: (token: string, approved: boolean) => void
  resolvingConfirm: boolean
  isLastUser?: boolean
}

export function MessageRow({
  chunk: c,
  tools,
  onEdit,
  onDelete,
  onRegen,
  onResync,
  onResolveConfirm,
  resolvingConfirm,
  isLastUser,
}: Props) {
  const { reasoning, body } = splitThink(c.content)
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

  const beginEdit = () => {
    setDraft(body)
    setEditing(true)
  }
  const saveEdit = () => {
    if (c.interaction_id != null) onEdit(c.interaction_id, draft)
    setEditing(false)
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

        {reasoning && <ReasoningFold reasoning={reasoning} />}

        {(c.tool_context || []).map((tc) => (
          <ToolCard key={tc.call_id} tc={tc} tools={tools} />
        ))}

        {editing ? (
          <div style={{ marginTop: 10 }}>
            <textarea
              className="editbox"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              autoFocus
            />
            <div className="editbar">
              <button className="btn approve" onClick={saveEdit}>
                save
              </button>
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
              {body}
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
          {c.role === 'assistant' && (
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
}
