import { useEffect, useRef, useState } from 'react'
import { getInfraStatus, runInfraAction } from '../api/client'
import { hasControlToken } from '../api/control_token'
import type {
  InfraAction,
  InfraGuest,
  InfraModel,
  InfraStatus,
} from '../types/contracts'

// DP-439 INFRA panel. Spans the chan+convo+insp grid columns like MemoryPanel.
// Every read/action is the existing proxmox tool, called through
// /api/v1/infra/* — this component holds no knowledge of the node beyond what
// those tools return.

const POLL_MS = 15_000

interface Pending {
  action: InfraAction
  text: string
}

function guestLabel(g: InfraGuest): string {
  return `${g.kind.toUpperCase()} ${g.vmid}${g.name ? ` (${g.name})` : ''}`
}

function guestConfirm(verb: 'start' | 'stop' | 'reboot', g: InfraGuest): string {
  const label = guestLabel(g)
  if (verb === 'start') return `Start ${label}?`
  const hostNote = ' If this guest hosts derpr, this portal goes down with it.'
  if (verb === 'stop') {
    return `Hard power-off ${label}? Anything unsaved in the guest is lost.${hostNote}`
  }
  return `Reboot ${label}?${hostNote}`
}

function modelConfirm(m: InfraModel): string {
  const cold =
    m.tier === 'cold'
      ? ' It is on the cold tier: it promotes first, which takes minutes.'
      : ''
  return `Switch :5001 to ${m.name}? :5001 serves nothing while the new model loads.${cold}`
}

const NODE_CONFIRM =
  'Reboot the whole Proxmox node? Every guest goes down — including the one hosting derpr, so this portal is unreachable until the node is back.'

function SectionError({ message }: { message?: string }) {
  return (
    <div className="errrow" style={{ padding: '8px 12px', borderRadius: 6, fontSize: 12 }}>
      {message || 'read failed'}
    </div>
  )
}

export function InfraPanel() {
  const [status, setStatus] = useState<InfraStatus | null>(null)
  const [readAt, setReadAt] = useState<Date | null>(null)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [pending, setPending] = useState<Pending | null>(null)
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(null)

  const operator = hasControlToken()

  // The poll loop lives inside the effect; `reload` is a handle to it for the
  // Refresh button and post-action reads. Each read schedules the next only
  // after it settles, so a slow node (list_models is many SSH round trips)
  // never stacks overlapping polls, and unmount stops it.
  const reload = useRef<() => void>(() => {})

  useEffect(() => {
    let alive = true
    let timer: number | undefined
    const load = async () => {
      window.clearTimeout(timer)
      setLoading(true)
      try {
        const s = await getInfraStatus()
        if (!alive) return
        setStatus(s)
        setReadAt(new Date())
        setErr('')
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e))
      } finally {
        if (alive) {
          setLoading(false)
          timer = window.setTimeout(() => void load(), POLL_MS)
        }
      }
    }
    reload.current = () => void load()
    void load()
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [])

  const ask = (action: InfraAction, text: string) => {
    setResult(null)
    setPending({ action, text })
  }

  const confirm = async () => {
    if (!pending) return
    const { action } = pending
    setPending(null)
    setBusy(true)
    try {
      const res = await runInfraAction(action)
      const ok = res.status === 'ok' || res.status === 'promoting'
      const detail =
        res.message ||
        (res.state ? `state: ${res.state}` : '') ||
        (typeof res.note === 'string' ? res.note : '') ||
        res.status
      setResult({ ok, text: `${action.action}: ${detail}` })
    } catch (e) {
      setResult({ ok: false, text: e instanceof Error ? e.message : String(e) })
    } finally {
      setBusy(false)
      reload.current()
    }
  }

  const pve = status?.pve
  const gpu = status?.gpu
  const models = status?.models
  const canAct = operator && !busy && !pending

  return (
    <div
      className="col convo infra"
      style={{ gridColumn: '2 / 5', overflow: 'auto', padding: '16px 20px' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <h2 style={{ margin: 0, fontSize: 15, letterSpacing: 0.5 }}>⌬ INFRA — Proxmox node</h2>
        <div className="grow" />
        <span style={{ fontSize: 11, opacity: 0.6 }}>
          {loading ? 'reading…' : readAt ? `read ${readAt.toLocaleTimeString()}` : ''}
        </span>
        <button className="btn" disabled={loading} onClick={() => reload.current()}>
          Refresh
        </button>
      </div>

      {err && (
        <div className="errrow" style={{ padding: '8px 12px', marginBottom: 12, borderRadius: 6 }}>
          {err}
        </div>
      )}

      {status && !status.enabled && (
        <div className="errrow" style={{ padding: '8px 12px', marginBottom: 12, borderRadius: 6 }}>
          Proxmox access is disabled on this instance (PVE_TOOLS_ENABLED is off, or no host key is
          mounted).
        </div>
      )}

      {!operator && (
        <div className="dimrow" style={{ marginBottom: 12, fontSize: 12 }}>
          Read-only: set the operator control token (top bar) to enable actions.
        </div>
      )}

      {pending && (
        <div className="infra-confirm">
          <div style={{ flex: 1 }}>{pending.text}</div>
          <button className="btn deny" onClick={() => void confirm()}>
            Confirm
          </button>
          <button className="btn" onClick={() => setPending(null)}>
            Cancel
          </button>
        </div>
      )}

      {result && (
        <div
          className={result.ok ? 'infra-result ok' : 'errrow'}
          style={{ padding: '8px 12px', marginBottom: 12, borderRadius: 6, fontSize: 12 }}
        >
          {result.text}
        </div>
      )}

      {/* ---- host strip ---- */}
      <section className="infra-strip">
        <div>
          <span className="infra-k">node</span>
          {pve?.status === 'ok' ? pve.uptime || '—' : '—'}
        </div>
        {gpu?.status === 'ok' &&
          gpu.gpus.map((g) => (
            <div key={g.card}>
              <span className="infra-k">{g.card} vram</span>
              {(g.vram_used_mib / 1024).toFixed(1)} / {(g.vram_total_mib / 1024).toFixed(1)} GiB
              <span style={{ opacity: 0.6 }}> · {(g.vram_free_mib / 1024).toFixed(1)} free</span>
            </div>
          ))}
        {gpu?.status === 'error' && (
          <div>
            <span className="infra-k">gpu</span>
            <span style={{ color: 'var(--danger)' }}>{gpu.message || 'unreadable'}</span>
          </div>
        )}
      </section>

      <div className="infra-grid">
        {/* ---- guests ---- */}
        <section>
          <div className="infra-h">GUESTS</div>
          {pve?.status === 'error' && <SectionError message={pve.message} />}
          {pve?.status === 'ok' && (
            <>
              <table className="infra-table">
                <tbody>
                  {pve.guests.map((g) => {
                    const running = g.status === 'running'
                    const act = (verb: 'start' | 'stop' | 'reboot') =>
                      ask(
                        { action: `${verb}_guest`, vmid: g.vmid, kind: g.kind },
                        guestConfirm(verb, g),
                      )
                    return (
                      <tr key={`${g.kind}${g.vmid}`}>
                        <td style={{ opacity: 0.7 }}>{g.kind}</td>
                        <td>{g.vmid}</td>
                        <td>{g.name || '—'}</td>
                        <td>
                          <span className={'pill' + (running ? ' infra-on' : '')}>{g.status}</span>
                          {g.lock && <span className="pill bindwarn">lock: {g.lock}</span>}
                        </td>
                        <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                          {running ? (
                            <>
                              <button className="btn" disabled={!canAct} onClick={() => act('reboot')}>
                                reboot
                              </button>{' '}
                              <button className="btn deny" disabled={!canAct} onClick={() => act('stop')}>
                                stop
                              </button>
                            </>
                          ) : (
                            <button className="btn approve" disabled={!canAct} onClick={() => act('start')}>
                              start
                            </button>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {pve.inventory_errors?.map((e) => (
                <div key={e} className="dimrow" style={{ color: 'var(--danger)' }}>
                  {e}
                </div>
              ))}
              <div style={{ marginTop: 14 }}>
                <button
                  className="btn deny"
                  disabled={!canAct}
                  onClick={() => ask({ action: 'reboot_node' }, NODE_CONFIRM)}
                >
                  reboot node…
                </button>
              </div>
            </>
          )}
        </section>

        {/* ---- models ---- */}
        <section>
          <div className="infra-h">MODELS on :5001</div>
          {models?.status === 'error' && <SectionError message={models.message} />}
          {models?.status === 'ok' && (
            <>
              {models.models.length === 0 && (
                <div className="dimrow">No switchable models reported.</div>
              )}
              <table className="infra-table">
                <tbody>
                  {models.models.map((m) => {
                    const active = m.state === 'active'
                    return (
                      <tr key={m.name}>
                        <td>
                          <span className={'infra-dot' + (active ? ' on' : '')} />
                          {m.name}
                        </td>
                        <td>
                          <span
                            className="pill"
                            title={
                              m.tier === 'cold'
                                ? 'cold archive tier: activating promotes it to the hot tier first, which takes minutes'
                                : 'hot tier: switches straight away'
                            }
                          >
                            {m.tier}
                          </span>
                          {m.pinned && <span className="pill">pinned</span>}
                        </td>
                        <td style={{ opacity: 0.7 }}>
                          {m.contextsize ? `${Math.round(m.contextsize / 1024)}k ctx` : ''}
                        </td>
                        <td style={{ opacity: 0.7 }}>{m.state}</td>
                        <td style={{ textAlign: 'right' }}>
                          {active ? (
                            <span style={{ color: 'var(--ok)', fontSize: 11 }}>active</span>
                          ) : (
                            <button
                              className="btn"
                              disabled={!canAct}
                              onClick={() =>
                                ask({ action: 'set_active_model', name: m.name }, modelConfirm(m))
                              }
                            >
                              activate
                            </button>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
