import { useCallback, useEffect, useRef, useState } from 'react'
import {
  listMemoryBanks,
  listMemoryDocuments,
  listMemoryOperations,
  deleteMemoryDocument,
  uploadMemoryFiles,
  ingestMemoryUrl,
  ingestMemoryPath,
} from '../api/client'
import { hasControlToken } from '../api/control_token'
import type {
  MemoryBank,
  MemoryDocument,
  MemoryOperation,
} from '../types/contracts'

// DP-292 import panel. Spans the chan+convo+insp grid columns (rail stays).
// Imports tab only for now; Inspector/recall tabs are future work.

function docKey(d: MemoryDocument): string {
  return String(d.document_id ?? d.id ?? '')
}

export function MemoryPanel() {
  const [banks, setBanks] = useState<MemoryBank[]>([])
  const [bank, setBank] = useState<string>('')
  const [docs, setDocs] = useState<MemoryDocument[]>([])
  const [ops, setOps] = useState<MemoryOperation[]>([])
  const [err, setErr] = useState<string>('')
  const [busy, setBusy] = useState(false)
  const [url, setUrl] = useState('')
  const [path, setPath] = useState('')
  const [glob, setGlob] = useState('**/*.md')
  const fileRef = useRef<HTMLInputElement>(null)

  const fail = (e: unknown) => setErr(e instanceof Error ? e.message : String(e))

  // First setState only AFTER an await, never synchronously in the effect —
  // satisfies react-hooks/set-state-in-effect (avoids cascading renders).
  const loadBanks = useCallback(async () => {
    try {
      const b = await listMemoryBanks()
      setErr('')
      setBanks(b)
      setBank((cur) => cur || (b[0]?.bank_id ?? ''))
    } catch (e) {
      fail(e)
    }
  }, [])

  const refresh = useCallback(async (b: string) => {
    if (!b) return
    try {
      const [d, o] = await Promise.all([
        listMemoryDocuments(b, { limit: 200 }),
        listMemoryOperations(b),
      ])
      setErr('')
      setDocs(d.items ?? [])
      setOps(o.operations ?? [])
    } catch (e) {
      fail(e)
    }
  }, [])

  useEffect(() => {
    // Loaders only setState after their awaited fetch resolves (never
    // synchronously) — the rule's static check can't see the await boundary.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadBanks()
  }, [loadBanks])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh(bank)
  }, [bank, refresh])

  const guardToken = (): boolean => {
    if (!hasControlToken()) {
      setErr('operator control token required for ingest/delete — set it in the top bar')
      return false
    }
    return true
  }

  const withBusy = async (fn: () => Promise<void>) => {
    setBusy(true)
    try {
      await fn()
    } catch (e) {
      fail(e)
    } finally {
      setBusy(false)
    }
  }

  const onUpload = () => {
    const files = fileRef.current?.files
    if (!files || files.length === 0 || !guardToken()) return
    void withBusy(async () => {
      const res = await uploadMemoryFiles(bank, Array.from(files), 'ingest,upload')
      const rejected = res.results.filter((r) => r.status === 'rejected')
      if (rejected.length) {
        setErr(`rejected: ${rejected.map((r) => `${r.file} (${r.reason})`).join(', ')}`)
      }
      if (fileRef.current) fileRef.current.value = ''
      await refresh(bank)
    })
  }

  const onIngestUrl = () => {
    if (!url.trim() || !guardToken()) return
    void withBusy(async () => {
      await ingestMemoryUrl(bank, url.trim())
      setUrl('')
      await refresh(bank)
    })
  }

  const onIngestPath = () => {
    if (!path.trim() || !guardToken()) return
    void withBusy(async () => {
      await ingestMemoryPath(bank, path.trim(), glob.trim() || undefined)
      setPath('')
      await refresh(bank)
    })
  }

  const onDelete = (d: MemoryDocument) => {
    const key = docKey(d)
    if (!key || !guardToken()) return
    if (!window.confirm(`Delete document "${key}" and its memory units?`)) return
    void withBusy(async () => {
      await deleteMemoryDocument(bank, key)
      await refresh(bank)
    })
  }

  return (
    <div
      className="col convo"
      style={{ gridColumn: '2 / 5', overflow: 'auto', padding: '16px 20px' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <h2 style={{ margin: 0, fontSize: 15, letterSpacing: 0.5 }}>◈ MEMORY — Imports</h2>
        <div className="grow" />
        <select
          className="field"
          value={bank}
          onChange={(e) => setBank(e.target.value)}
          style={{ minWidth: 180 }}
        >
          {banks.length === 0 && <option value="">— no banks —</option>}
          {banks.map((b) => (
            <option key={b.bank_id} value={b.bank_id}>
              {b.bank_id}
              {b.fact_count != null ? ` (${b.fact_count})` : ''}
            </option>
          ))}
        </select>
        <button className="btn" disabled={busy} onClick={() => void refresh(bank)}>
          Refresh
        </button>
      </div>

      {err && (
        <div
          className="errrow"
          style={{ padding: '8px 12px', marginBottom: 12, borderRadius: 6 }}
        >
          {err}
        </div>
      )}

      {!hasControlToken() && (
        <div className="dimrow" style={{ marginBottom: 12, fontSize: 12 }}>
          Read-only: set the operator control token (top bar) to enable ingest/delete.
        </div>
      )}

      {/* ---- ingest sources ---- */}
      <section style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>ADD DOCUMENTS</div>
        <div style={{ display: 'grid', gap: 10, maxWidth: 720 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input ref={fileRef} type="file" accept=".md,.txt" multiple className="field" />
            <button className="btn" disabled={busy || !bank} onClick={onUpload}>
              Upload (.md/.txt)
            </button>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              className="field"
              placeholder="https://example.com/doc.md"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              style={{ flex: 1 }}
            />
            <button className="btn" disabled={busy || !bank} onClick={onIngestUrl}>
              Fetch URL
            </button>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              className="field"
              placeholder="server path (file or dir)"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              style={{ flex: 2 }}
            />
            <input
              className="field"
              placeholder="glob"
              value={glob}
              onChange={(e) => setGlob(e.target.value)}
              style={{ flex: 1 }}
            />
            <button className="btn" disabled={busy || !bank} onClick={onIngestPath}>
              Ingest path
            </button>
          </div>
        </div>
      </section>

      {/* ---- documents ---- */}
      <section style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>
          DOCUMENTS ({docs.length})
        </div>
        {docs.length === 0 ? (
          <div className="dimrow" style={{ fontSize: 12 }}>No documents in this bank.</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ textAlign: 'left', opacity: 0.6 }}>
                <th style={{ padding: '4px 8px' }}>document</th>
                <th style={{ padding: '4px 8px' }}>units</th>
                <th style={{ padding: '4px 8px' }}>updated</th>
                <th style={{ padding: '4px 8px' }} />
              </tr>
            </thead>
            <tbody>
              {docs.map((d) => (
                <tr key={docKey(d)} style={{ borderTop: '1px solid var(--line)' }}>
                  <td style={{ padding: '4px 8px', wordBreak: 'break-all' }}>{docKey(d)}</td>
                  <td style={{ padding: '4px 8px' }}>{d.memory_unit_count ?? '—'}</td>
                  <td style={{ padding: '4px 8px', opacity: 0.7 }}>
                    {d.updated_at ? d.updated_at.slice(0, 19).replace('T', ' ') : '—'}
                  </td>
                  <td style={{ padding: '4px 8px', textAlign: 'right' }}>
                    <button
                      className="ibtn"
                      disabled={busy}
                      onClick={() => onDelete(d)}
                      title="delete document"
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* ---- operations monitor ---- */}
      <section>
        <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>
          OPERATIONS ({ops.length})
        </div>
        {ops.length === 0 ? (
          <div className="dimrow" style={{ fontSize: 12 }}>No recent operations.</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <tbody>
              {ops.map((o) => (
                <tr key={o.id} style={{ borderTop: '1px solid var(--line)' }}>
                  <td style={{ padding: '4px 8px' }}>
                    <span className="pill">{o.status}</span>
                  </td>
                  <td style={{ padding: '4px 8px', opacity: 0.8 }}>{o.task_type ?? '—'}</td>
                  <td style={{ padding: '4px 8px', wordBreak: 'break-all', opacity: 0.7 }}>
                    {o.document_id ?? ''}
                  </td>
                  <td style={{ padding: '4px 8px', color: 'var(--err, #e66)' }}>
                    {o.error_message ?? ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
