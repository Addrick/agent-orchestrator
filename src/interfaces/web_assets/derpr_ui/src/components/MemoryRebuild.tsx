import { useCallback, useEffect, useRef, useState } from 'react'
import { getMemoryRebuild, startMemoryRebuild } from '../api/client'
import { hasControlToken } from '../api/control_token'
import type { MemoryRebuildPersona, MemoryRebuildState } from '../types/contracts'

// DP-506: "purge Hindsight and requeue all persona history from SQL".
// Destructive: every memory persona's bank is deleted outright. The dialog
// names the banks and what is lost, and the run starts only after the
// operator types the phrase the server also demands.

const PHRASE = 'PURGE'
const POLL_MS = 3000

function describe(p: MemoryRebuildPersona): string {
  const c = p.counts
  if (p.state === 'error') return p.error ?? 'failed'
  if (p.state === 'waiting') return 'waiting'
  if (p.state === 'skipped') return 'not touched: the run stopped at an earlier error'
  const sent = (c.blocks ?? 0) + (c.queued ?? 0)
  const parts = [`${sent}/${c.total ?? '?'} blocks`]
  if (c.adopted) parts.push(`${c.adopted} from old history`)
  if (c.pending) parts.push(`${c.pending} turns kept in prompt`)
  return parts.join(' · ')
}

export function MemoryRebuild({ onFinished }: { onFinished: () => void }) {
  const [job, setJob] = useState<MemoryRebuildState | null>(null)
  const [open, setOpen] = useState(false)
  const [typed, setTyped] = useState('')
  const [err, setErr] = useState('')
  const running = job?.status === 'running'

  const load = useCallback(async () => {
    try {
      setJob(await getMemoryRebuild())
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    // setState happens only after the awaited fetch resolves.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load()
  }, [load])

  useEffect(() => {
    if (!running) return
    const t = window.setInterval(() => void load(), POLL_MS)
    return () => window.clearInterval(t)
  }, [running, load])

  // Refresh the panel once, on the running -> finished transition only — not
  // on unmount or whenever the parent hands in a new callback.
  const wasRunning = useRef(false)
  useEffect(() => {
    if (wasRunning.current && !running) onFinished()
    wasRunning.current = running
  }, [running, onFinished])

  const close = () => {
    setOpen(false)
    setTyped('')
  }

  const start = async () => {
    if (typed !== PHRASE) return
    try {
      setErr('')
      setJob(await startMemoryRebuild(typed))
      close()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const targets = job?.targets ?? []

  return (
    <section style={{ marginTop: 28, paddingTop: 14, borderTop: '1px solid var(--line)' }}>
      <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>DANGER ZONE</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button
          className="btn deny"
          disabled={running || !hasControlToken() || targets.length === 0}
          onClick={() => setOpen(true)}
          title={hasControlToken() ? '' : 'operator control token required'}
        >
          Purge &amp; rebuild all persona memory
        </button>
        <span className="dimrow" style={{ fontSize: 12 }}>
          Deletes every memory persona's Hindsight bank and resends its chat history from the database.
        </span>
      </div>

      {err && (
        <div className="errrow" style={{ padding: '8px 12px', marginTop: 10, borderRadius: 6 }}>
          {err}
        </div>
      )}

      {job && job.personas.length > 0 && (
        <table style={{ marginTop: 10, borderCollapse: 'collapse', fontSize: 13 }}>
          <tbody>
            {job.personas.map((p) => (
              <tr key={p.name} style={{ borderTop: '1px solid var(--line)' }}>
                <td style={{ padding: '4px 8px' }}>{p.name}</td>
                <td style={{ padding: '4px 8px' }}>
                  <span className="pill">{p.state}</span>
                </td>
                <td
                  style={{
                    padding: '4px 8px',
                    opacity: 0.8,
                    color: p.state === 'error' ? 'var(--danger)' : undefined,
                  }}
                >
                  {describe(p)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {job?.status === 'finished' && (
        <div className="dimrow" style={{ fontSize: 12, marginTop: 6 }}>
          Sends finished. Hindsight keeps extracting in the background; the operations monitor shows its progress.
        </div>
      )}

      {open && (
        <div className="modal-backdrop" onClick={close}>
          <div
            className="modal"
            role="alertdialog"
            aria-modal="true"
            aria-label="confirm memory purge"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === 'Escape') close()
            }}
          >
            <div className="modal-head">
              <b style={{ color: 'var(--danger)' }}>⚠ Are you absolutely sure?</b>
              <button className="cbtn" title="close" onClick={close}>
                ✕
              </button>
            </div>
            <div className="modal-body" style={{ fontSize: 13, lineHeight: 1.5 }}>
              <p style={{ marginTop: 0 }}>
                This <b>permanently deletes</b> these {targets.length} Hindsight bank(s):
              </p>
              <p>
                <b>{targets.join(', ')}</b>
              </p>
              <p>
                Every fact, observation and mental model goes, and so does every document added
                from this panel or by a script (uploads, URLs, server paths, Claude Code session
                ingests). <b>Those do not come back.</b> Only chat history from the database is
                resent, and Hindsight then has to extract all of it again, which can take hours on
                a local model. Recall stays incomplete until it finishes.
              </p>
              <p>
                Type <b>{PHRASE}</b> to confirm.
              </p>
              <input
                className="field"
                autoFocus
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') void start()
                }}
                style={{ width: '100%' }}
              />
            </div>
            <div className="savebar" style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', padding: 10 }}>
              <button className="btn" onClick={close}>
                Cancel
              </button>
              <button className="btn deny" disabled={typed !== PHRASE} onClick={() => void start()}>
                Delete banks and rebuild
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
