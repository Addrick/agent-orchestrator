/* DP-438 — switch which server `model local` talks to, from the statusline.
 *
 * Apply is temporary (lost on engine restart); Save persists the applied URL;
 * Reset drops both and returns to LOCAL_LLM_URL. Turns already streaming finish
 * on the old server. The three writes need the operator token.
 */
import { useEffect, useState } from 'react'
import {
  applyLocalBackend,
  getLocalBackend,
  resetLocalBackend,
  saveLocalBackend,
} from '../api/client'
import type { LocalBackendState } from '../types/contracts'

/** `label` when the URL matches a preset, else host:port. */
function shortName(s: LocalBackendState): string {
  if (s.label) return s.label
  try {
    return new URL(s.url).host
  } catch {
    return s.url
  }
}

const SOURCE_HINT: Record<LocalBackendState['source'], string> = {
  temporary: 'applied, not saved — reverts on engine restart',
  saved: 'saved — survives restarts',
  env: 'from LOCAL_LLM_URL',
  default: 'config default (LOCAL_LLM_URL unset)',
}

export function LocalBackendSwitch({ onSwitched }: { onSwitched: () => void }) {
  const [state, setState] = useState<LocalBackendState | null>(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    getLocalBackend().then(setState).catch(() => setState(null))
  }, [])

  if (!state) return null
  return (
    <>
      <button
        className="sl-chip sl-switch"
        title={`local model server: ${state.url} (${SOURCE_HINT[state.source]}) — click to switch`}
        onClick={() => setOpen(true)}
      >
        ⇄ <b>{shortName(state)}</b>
        {state.source === 'temporary' && <span className="sl-sub">· unsaved</span>}
      </button>
      {open && (
        <LocalBackendModal
          state={state}
          onChange={(s, switched) => {
            setState(s)
            if (switched) onSwitched()
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  )
}

function LocalBackendModal({
  state,
  onChange,
  onClose,
}: {
  state: LocalBackendState
  onChange: (s: LocalBackendState, switched: boolean) => void
  onClose: () => void
}) {
  const [url, setUrl] = useState(state.url)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async (op: () => Promise<LocalBackendState>) => {
    setBusy(true)
    setError(null)
    try {
      const next = await op()
      onChange(next, next.url !== state.url)
      setUrl(next.url)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  const apply = () => run(() => applyLocalBackend(url))

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="local model server"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === 'Escape') onClose()
        }}
      >
        <div className="modal-head">
          <b>Local model server</b>
          <button className="cbtn" title="close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="field">
            <span className="lbl">active · {state.source}</span>
            <span className="modal-hint" style={{ marginTop: 0 }}>
              <b>{state.url}</b> — {SOURCE_HINT[state.source]}
            </span>
          </div>
          <div className="field">
            <span className="lbl">url</span>
            <div className="ctrl">
              <input
                autoFocus
                value={url}
                placeholder="http://10.0.0.72:5001"
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !busy) apply()
                }}
              />
            </div>
            {state.presets.length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                {state.presets.map((p) => (
                  <button
                    key={p.url}
                    className="mini"
                    title={p.url}
                    aria-pressed={p.url === state.url}
                    onClick={() => setUrl(p.url)}
                  >
                    {p.label || p.url}
                  </button>
                ))}
              </div>
            )}
            <span className="modal-hint">
              Apply switches now, until the engine restarts. Save keeps it across restarts. Replies already
              streaming finish on the old server.
            </span>
          </div>
        </div>

        <div className="savebar">
          {error && <span className="savestatus err">{error}</span>}
          <span className="grow" />
          <button
            className="mini"
            title="drop the applied and saved server; use LOCAL_LLM_URL"
            onClick={() => run(resetLocalBackend)}
            disabled={busy || state.source === 'env' || state.source === 'default'}
          >
            reset
          </button>
          <button
            className="mini"
            title="keep the applied server across restarts"
            onClick={() => run(saveLocalBackend)}
            disabled={busy || state.source !== 'temporary'}
          >
            save
          </button>
          <button className="savebtn" onClick={apply} disabled={busy || !url.trim()}>
            {busy ? '…' : 'apply'}
          </button>
        </div>
      </div>
    </div>
  )
}
