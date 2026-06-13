import type { StreamState } from '../state/store'
import type { ToolDef } from '../types/contracts'
import { splitThink } from '../state/util'
import { ReasoningFold } from './ReasoningFold'
import { ToolCard } from './ToolCard'
import { deriveTreatment } from '../state/util'

interface Props {
  stream: StreamState
  tools: ToolDef[]
  onDismiss: () => void
  onRegen: () => void
}

/** The in-flight assistant turn + terminal error/aborted treatments.
 *  This is transient UI; on a clean [DONE] the parent re-fetches the
 *  transcript and clears this, so the persisted row replaces it. */
export function StreamRow({ stream: s, tools, onDismiss, onRegen }: Props) {
  if (!s.active && !s.errored && !s.aborted) return null

  const treatment = deriveTreatment({
    responseType: s.responseType || undefined,
    aborted: s.aborted,
    errored: s.errored,
    hadTools: s.tools.length > 0,
    emptyContent: !s.text.trim(),
  })

  if (treatment === 'error') {
    return (
      <div className="errrow">
        <div className="lh">
          <span className="lbl">⚠ error · turn failed</span>
        </div>
        <div className="text">{s.errorMsg || 'engine/provider failure'}</div>
        <div className="text" style={{ color: 'var(--ink-faint)', marginTop: 6 }}>
          No assistant row was committed.
        </div>
        <div className="acts">
          <button className="btn" onClick={onDismiss}>
            dismiss
          </button>
          <button className="btn approve" onClick={onRegen}>
            retry
          </button>
        </div>
      </div>
    )
  }

  const { reasoning, body } = splitThink(s.text)
  // No frame of any kind yet — the provider hasn't produced bytes. True for
  // the whole wait on non-streaming providers (everything except local), so
  // show an explicit typing indicator rather than a bare cursor.
  const waiting = s.active && !s.text && s.tools.length === 0

  return (
    <div className={'msg' + (treatment === 'aborted' ? ' aborted' : '')}>
      <div className="gut">
        <div className="av assistant">AS</div>
      </div>
      <div className="bd">
        <div className="meta">
          <span className="who assistant">assistant</span>
          <span className="idtag">{s.active ? 'streaming…' : 'pending id'}</span>
          {treatment === 'tool-only' && (
            <span className="chip toolonly-chip">tool-only turn</span>
          )}
        </div>

        {reasoning && <ReasoningFold reasoning={reasoning} />}

        {s.tools.map((tc) => (
          <ToolCard key={tc.call_id} tc={tc} tools={tools} />
        ))}

        {waiting ? (
          <div className="text typing">
            <span className="typing-dots">
              <span />
              <span />
              <span />
            </span>
            waiting for model…
          </div>
        ) : (
          (body.trim() || s.active) && (
            <div
              className="text"
              style={{ marginTop: reasoning || s.tools.length ? 10 : 0 }}
            >
              {body}
              {s.active && <span className="stream-cursor" />}
            </div>
          )
        )}

        {treatment === 'aborted' && (
          <div className="abortmark">aborted · partial flushed · regen to continue</div>
        )}
      </div>
    </div>
  )
}
