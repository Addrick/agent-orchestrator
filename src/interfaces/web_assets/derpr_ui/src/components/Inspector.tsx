import { useMemo, useState } from 'react'
import type { Persona, ToolDef } from '../types/contracts'
import type { PortalStore } from '../state/store'
import { policyLabel } from '../state/util'

type Tab = 'persona' | 'tools' | 'raw'

interface Props {
  store: PortalStore
}

// Inspector chrome. Persona tab = editable base-vs-kobold split (PATCH for the
// fields the adapter accepts, dev_command for thinking_level); Tools tab = live
// enable toggles via `set tools`; Raw req = the parity /assemble tab (S5).
export function Inspector({ store }: Props) {
  const [tab, setTab] = useState<Tab>('persona')
  const { persona, tools } = store

  return (
    <div className="col insp">
      <div className="insp-tabs">
        {(['persona', 'tools', 'raw'] as Tab[]).map((t) => (
          <button
            key={t}
            className={'insp-tab' + (tab === t ? ' active' : '')}
            onClick={() => setTab(t)}
          >
            {t === 'raw' ? 'Raw req' : t}
          </button>
        ))}
      </div>
      <div className="insp-body">
        {/* key on persona name so switching personas remounts the editable
            buffer fresh; a same-persona refetch keeps typed edits and just
            re-derives dirtiness from the new baseline. */}
        {tab === 'persona' && <PersonaPane key={persona?.name ?? '∅'} store={store} />}
        {tab === 'tools' && <ToolsPane persona={persona} tools={tools} store={store} />}
        {tab === 'raw' && <RawPane />}
      </div>
    </div>
  )
}

// ---- editable persona buffer ----------------------------------------------
// The buffer holds every editable field as a string (inputs yield strings; an
// empty kobold-extra string means "clear"). Diffing the stringified buffer vs.
// the persona tells us exactly which keys changed, so the PATCH carries only
// real edits and the dev_command only fires when thinking_level moved.

// Memory modes accepted by the engine (Persona.set_memory_mode).
const MEMORY_MODES = [
  'CHANNEL_ISOLATED',
  'SERVER_WIDE',
  'PERSONAL',
  'GLOBAL',
  'TICKET_ISOLATED',
]

// PATCH-able base params that are plain strings on the wire.
const STR_BASE: (keyof Persona)[] = ['model_name', 'chat_template']
// PATCH-able base params coerced to numbers by the adapter's setters.
const NUM_BASE: (keyof Persona)[] = [
  'temperature',
  'top_p',
  'top_k',
  'max_tokens',
  'history_messages',
  'max_context_tokens',
]
// PATCH-able kobold sampler extras (top-level keys on the PATCH body; the
// adapter routes them into provider_extras["kobold"]). Empty string clears.
const KOBOLD_EXTRAS = ['rep_pen', 'rep_pen_range', 'rep_pen_slope', 'min_p', 'typical', 'tfs']

type Buf = Record<string, string>

function asStr(v: unknown): string {
  return v == null ? '' : String(v)
}

function buildBuffer(p: Persona): Buf {
  const b: Buf = {
    prompt: asStr(p.prompt),
    memory_mode: asStr(p.memory_mode),
    thinking_level: asStr(p.thinking_level),
  }
  for (const k of STR_BASE) b[k] = asStr(p[k])
  for (const k of NUM_BASE) b[k] = asStr(p[k])
  const kx = p.kobold_extras || {}
  for (const k of KOBOLD_EXTRAS) b[k] = asStr(kx[k])
  return b
}

// Number if it parses cleanly; otherwise the raw string (the adapter will
// coerce/reject and report it back in rejected_fields).
function coerceNum(s: string): number | string {
  const n = Number(s)
  return s.trim() !== '' && !Number.isNaN(n) ? n : s
}

function PersonaPane({ store }: { store: PortalStore }) {
  const p = store.persona
  const modelList = store.modelList
  const [koboldCollapsed, setKoboldCollapsed] = useState(true)
  const [buf, setBuf] = useState<Buf>(() => (p ? buildBuffer(p) : {}))
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState<
    { kind: 'ok' | 'warn' | 'err'; text: string } | null
  >(null)

  // baseline tracks the canonical persona; after a save → refetch the persona
  // object changes identity, so dirtiness re-derives against the saved values
  // without resetting the user's buffer.
  const baseline = useMemo(() => (p ? buildBuffer(p) : {}), [p])
  const dirty = useMemo(
    () => Object.keys(buf).some((k) => buf[k] !== baseline[k]),
    [buf, baseline],
  )

  if (!p) return <div className="dimrow">no persona loaded</div>

  const set = (k: string, v: string) => setBuf((b) => ({ ...b, [k]: v }))
  const reset = () => {
    setBuf(buildBuffer(p))
    setStatus(null)
  }

  const onSave = async () => {
    const patch: Record<string, unknown> = {}
    // prompt + string base params
    if (buf.prompt !== baseline.prompt) patch.prompt = buf.prompt
    for (const k of STR_BASE) if (buf[k] !== baseline[k]) patch[k] = buf[k]
    if (buf.memory_mode !== baseline.memory_mode) patch.memory_mode = buf.memory_mode
    // numeric base params
    for (const k of NUM_BASE) {
      if (buf[k] !== baseline[k]) patch[k] = buf[k] === '' ? null : coerceNum(buf[k])
    }
    // kobold sampler extras — empty string clears the entry server-side
    for (const k of KOBOLD_EXTRAS) {
      if (buf[k] !== baseline[k]) patch[k] = buf[k] === '' ? '' : coerceNum(buf[k])
    }
    // thinking_level is not a PATCH key → dev_command
    const devCommands: string[] = []
    if (buf.thinking_level !== baseline.thinking_level) {
      const lvl = buf.thinking_level.trim()
      devCommands.push(`set thinking_level ${lvl === '' ? 'none' : lvl}`)
    }

    setSaving(true)
    setStatus(null)
    try {
      const res = await store.savePersona(patch, devCommands)
      const problems = [...(res.rejected_fields || []), ...(res.unknown_fields || [])]
      if (res.error) {
        setStatus({ kind: 'err', text: `${res.error}: ${res.detail || 'save failed'}` })
      } else if (problems.length) {
        setStatus({
          kind: 'warn',
          text: `saved, but ignored/coerced: ${problems.join(', ')}`,
        })
      } else {
        setStatus({ kind: 'ok', text: 'saved' })
      }
    } catch (e) {
      setStatus({ kind: 'err', text: `save failed: ${e}` })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="insp-pane active">
      {p.security_blocked && (
        <div className="secbanner" style={{ display: 'block' }}>
          ⚠ persona security-blocked —{' '}
          {(p.security_block_reasons || []).join('; ') || 'tooling disabled'}
        </div>
      )}

      <Single label="identity">
        <span>{p.name}</span>
        <span style={{ color: 'var(--ink-faint)' }}>{p.display_name}</span>
      </Single>

      <div className="field">
        <span className="lbl">system prompt</span>
        <textarea
          className="ctrl area edit"
          value={buf.prompt}
          onChange={(e) => set('prompt', e.target.value)}
        />
      </div>

      <div className="section">
        ▣ base params<span className="desc">sent to every provider</span>
      </div>

      <Pair>
        <Cell label="model_name">
          {/* same catalog as the `what models` dev command (models_available) */}
          <select value={buf.model_name} onChange={(e) => set('model_name', e.target.value)}>
            {buf.model_name && !modelList.includes(buf.model_name) && (
              <option value={buf.model_name}>{buf.model_name}</option>
            )}
            {modelList.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Cell>
        <Cell label="memory_mode">
          <select value={buf.memory_mode} onChange={(e) => set('memory_mode', e.target.value)}>
            {!MEMORY_MODES.includes(buf.memory_mode) && buf.memory_mode && (
              <option value={buf.memory_mode}>{buf.memory_mode}</option>
            )}
            {MEMORY_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Cell>
      </Pair>

      <div className="field">
        <span className="lbl">temperature · {buf.temperature || '—'}</span>
        <div className="slider">
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={buf.temperature === '' ? 0 : Number(buf.temperature)}
            onChange={(e) => set('temperature', e.target.value)}
          />
          <span className="val">{buf.temperature === '' ? '—' : Number(buf.temperature).toFixed(2)}</span>
        </div>
      </div>

      <Pair>
        <Cell label="max_tokens">
          <input type="number" value={buf.max_tokens} onChange={(e) => set('max_tokens', e.target.value)} />
        </Cell>
        <Cell label="history_messages">
          <input
            type="number"
            value={buf.history_messages}
            onChange={(e) => set('history_messages', e.target.value)}
          />
        </Cell>
      </Pair>

      <Pair>
        <Cell label="max_context_tokens">
          <input
            type="number"
            value={buf.max_context_tokens}
            onChange={(e) => set('max_context_tokens', e.target.value)}
          />
        </Cell>
        <Cell label="thinking_level · dev_command">
          <input
            value={buf.thinking_level}
            placeholder="none"
            onChange={(e) => set('thinking_level', e.target.value)}
          />
        </Cell>
      </Pair>

      <Pair>
        <Cell label="chat_template">
          <input value={buf.chat_template} onChange={(e) => set('chat_template', e.target.value)} />
        </Cell>
        <Cell label="tool_policy · read-only">
          <span title="edit via the Tools tab / `set tool_policy`">
            {policyLabel(p.tool_policy)}
          </span>
        </Cell>
      </Pair>

      <div
        className={'section kobold' + (koboldCollapsed ? ' collapsed' : '')}
        onClick={() => setKoboldCollapsed((c) => !c)}
      >
        ⚠ kobold samplers<span className="pill">local-model only</span>
        <span className="desc">provider_extra · engine forwards these to local models</span>
        <span className="car">▾</span>
      </div>
      <div className={'kobold-fields' + (koboldCollapsed ? ' collapsed' : '')}>
        <Pair>
          <Cell label="top_p">
            <input type="number" step="0.01" value={buf.top_p} onChange={(e) => set('top_p', e.target.value)} />
          </Cell>
          <Cell label="top_k">
            <input type="number" value={buf.top_k} onChange={(e) => set('top_k', e.target.value)} />
          </Cell>
        </Pair>
        <Pair>
          <Cell label="rep_pen">
            <input type="number" step="0.01" value={buf.rep_pen} onChange={(e) => set('rep_pen', e.target.value)} />
          </Cell>
          <Cell label="rep_pen_range">
            <input
              type="number"
              value={buf.rep_pen_range}
              onChange={(e) => set('rep_pen_range', e.target.value)}
            />
          </Cell>
        </Pair>
        <Pair>
          <Cell label="min_p">
            <input type="number" step="0.01" value={buf.min_p} onChange={(e) => set('min_p', e.target.value)} />
          </Cell>
          <Cell label="tfs">
            <input type="number" step="0.01" value={buf.tfs} onChange={(e) => set('tfs', e.target.value)} />
          </Cell>
        </Pair>
        <Pair>
          <Cell label="rep_pen_slope">
            <input
              type="number"
              step="0.01"
              value={buf.rep_pen_slope}
              onChange={(e) => set('rep_pen_slope', e.target.value)}
            />
          </Cell>
          <Cell label="typical">
            <input type="number" step="0.01" value={buf.typical} onChange={(e) => set('typical', e.target.value)} />
          </Cell>
        </Pair>
        {/* mirostat / sampler_order / instruct_tags are not in the adapter's
            PATCH coercion set — shown read-only until a dedicated editor lands. */}
        <div className="field">
          <span className="lbl">mirostat · tau · eta · sampler_order · instruct_tags (read-only)</span>
          <div className="ctrl">
            <span>
              {asStr(p.kobold_extras?.mirostat) || '—'} · {asStr(p.kobold_extras?.mirostat_tau) || '—'} ·{' '}
              {asStr(p.kobold_extras?.mirostat_eta) || '—'} · [{(p.kobold_extras?.sampler_order || []).join(', ')}] ·{' '}
              {p.instruct_tags && Object.keys(p.instruct_tags).length ? 'custom' : '—'}
            </span>
          </div>
        </div>
      </div>

      <div className="savebar">
        {status && <span className={'savestatus ' + status.kind}>{status.text}</span>}
        <span className="grow" />
        <button className="mini" onClick={reset} disabled={!dirty || saving}>
          reset
        </button>
        <button className="savebtn" onClick={onSave} disabled={!dirty || saving}>
          {saving ? 'saving…' : 'save'}
        </button>
      </div>
    </div>
  )
}

// ---- small editable-field primitives --------------------------------------
function Single({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <span className="lbl">{label}</span>
      <div className="ctrl">{children}</div>
    </div>
  )
}

function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <span className="lbl">{label}</span>
      <div className="ctrl">{children}</div>
    </div>
  )
}

function Pair({ children }: { children: React.ReactNode }) {
  return (
    <div className="field">
      <div className="row2">{children}</div>
    </div>
  )
}

// ---- Tools tab ------------------------------------------------------------
// Mirrors the DP-120 Lite modal command builder: derive a `set tools …` string
// from the desired selection, then POST it through dev_command.
function buildToolsCommand(allNames: string[], selected: string[]): string {
  if (selected.length === 0) return 'set tools none'
  if (selected.length === allNames.length) return 'set tools all'
  if (selected.length > allNames.length / 2) {
    const excluded = allNames.filter((n) => !selected.includes(n))
    return 'set tools all ' + excluded.map((n) => '-' + n).join(' ')
  }
  return 'set tools ' + selected.join(' ')
}

function ToolsPane({
  persona,
  tools,
  store,
}: {
  persona: Persona | null
  tools: ToolDef[]
  store: PortalStore
}) {
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<{ kind: 'ok' | 'warn'; text: string } | null>(null)

  const allNames = tools.map((t) => t.name)
  const wildcard = (persona?.enabled_tools || []).includes('*')
  const enabled = new Set(wildcard ? allNames : persona?.enabled_tools || [])

  const onToggle = async (name: string) => {
    if (!persona || busy) return
    const next = new Set(enabled)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    const command = buildToolsCommand(allNames, [...next])
    setBusy(true)
    setNote(null)
    try {
      const resp = await store.runToolsCommand(command)
      if (resp.mutated) {
        setNote({ kind: 'ok', text: resp.response || 'updated' })
      } else {
        setNote({ kind: 'warn', text: resp.response || 'no change' })
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="insp-pane active">
      {persona?.security_blocked && (
        <div className="secbanner" style={{ display: 'block' }}>
          ⚠ security-blocked — generation refused until tools are scoped to a safe set
        </div>
      )}
      {note && <div className={'toolnote ' + note.kind}>{note.text}</div>}
      {tools.map((t) => {
        const caps = t.capabilities
        const on = enabled.has(t.name)
        return (
          <div className="toolrow" key={t.name}>
            <div className="tt">
              <span className="nm">{t.name}</span>
              <button
                className={'en' + (on ? ' on' : '')}
                title={on ? 'enabled — click to disable' : 'disabled — click to enable'}
                disabled={busy}
                onClick={() => onToggle(t.name)}
              >
                <span className="sw" />
              </button>
            </div>
            <div className="ds">{t.description}</div>
            <div className="tags">
              {t.is_write ? (
                <span className="badge write">write</span>
              ) : (
                <span className="badge read">read</span>
              )}
              <span
                className={
                  'badge ' +
                  (caps.sensitivity === 'high'
                    ? 'high'
                    : caps.sensitivity === 'medium'
                      ? 'med'
                      : 'low')
                }
              >
                {caps.sensitivity}
              </span>
              <span className={'badge ' + caps.locality}>{caps.locality}</span>
              {caps.produces_untrusted && <span className="badge low">untrusted</span>}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function RawPane() {
  return (
    <div className="insp-pane raw active">
      <div className="parity warn">
        <span className="pi warn">⚠ parity inspector — S5</span>
        <span className="pd">
          The exact assembled request, proven to match{' '}
          <code>chat_system.stream_response</code>, is sourced from the proposed{' '}
          <code>/assemble</code> dry-run endpoint (Sprint S5). Until that backend
          builder ships, this pane intentionally shows no reconstructed request to
          avoid implying a parity guarantee it can't make.
        </span>
      </div>
    </div>
  )
}
