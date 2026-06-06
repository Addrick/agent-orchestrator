import { useEffect, useMemo, useState } from 'react'
import type { Persona, ToolDef } from '../types/contracts'
import type { PortalStore } from '../state/store'
import { policyLabel, estimateTokens, fmtTok } from '../state/util'

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
        {tab === 'tools' && (
          <ToolsPane key={persona?.name ?? '∅'} persona={persona} tools={tools} store={store} />
        )}
        {tab === 'raw' && <RawPane store={store} />}
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
        <Cell label="tool_policy">
          <span title="edit in the Tools tab (default · allow/ask · overrides)">
            {policyLabel(p.tool_policy)} · edit in Tools ›
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

// ---- Tools tab — structured tool_policy editor ----------------------------
// The full policy (default + per-tool allow/ask + security overrides) is edited
// here and saved as ONE `set tool_policy <json>` dev_command — the engine path
// that revalidates security (message_handler `_handle_set`, DP-128). `set tools`
// could only express the allow-list; `ask` and `explicit_overrides` need the
// JSON setter. Persona refetch re-derives the baseline so dirtiness clears.

type ToolState = 'off' | 'allow' | 'ask'

// catalog group key for tools with no owning service (built-in / local)
const BUILTIN_GROUP = 'built-in'

// The three security-invariant escape hatches in policy.py:validate_composition.
const OVERRIDE_FLAGS: { key: string; label: string }[] = [
  { key: 'network_read_local_write', label: 'network:read + local:write' },
  { key: 'untrusted_read_network_write', label: 'untrusted:read + network:write' },
  { key: 'pii_read_network_any', label: 'pii:read + network:* egress' },
]

interface PolicyDraft {
  default: string // 'deny' | 'allow'
  states: Record<string, ToolState>
  overrides: Set<string>
  // service bindings the persona is granted (gate 2). A service group's tools
  // are inert until its binding is listed here, regardless of allow/ask.
  bindings: Set<string>
  // wildcard allow=['*'] is preserved verbatim until the user edits the toolset,
  // so an "allow everything (incl. future tools)" policy isn't silently frozen
  // into the current catalog.
  wildcard: boolean
}

function deriveDraft(persona: Persona | null, tools: ToolDef[]): PolicyDraft {
  const tp = persona?.tool_policy || {}
  const def = (tp.default as string) || 'deny'
  const allow = new Set(tp.allow || [])
  const ask = new Set(tp.ask || [])
  const wildcard = def === 'allow' && allow.has('*')
  const states: Record<string, ToolState> = {}
  for (const t of tools) {
    if (ask.has(t.name)) states[t.name] = 'ask'
    else if (wildcard || allow.has(t.name)) states[t.name] = 'allow'
    else states[t.name] = 'off'
  }
  return {
    default: def,
    states,
    overrides: new Set(tp.explicit_overrides || []),
    bindings: new Set(persona?.service_bindings || []),
    wildcard,
  }
}

// Build the tool_policy dict the engine expects. `capabilities_required` is
// passed through untouched (not editable here).
function draftToPolicy(
  draft: PolicyDraft,
  tools: ToolDef[],
  persona: Persona | null,
): Record<string, unknown> {
  const allTouchedAllowed =
    draft.default === 'allow' &&
    draft.wildcard &&
    tools.every((t) => draft.states[t.name] === 'allow')
  const allow = allTouchedAllowed
    ? ['*']
    : tools.filter((t) => draft.states[t.name] === 'allow').map((t) => t.name)
  const ask = tools.filter((t) => draft.states[t.name] === 'ask').map((t) => t.name)
  return {
    default: draft.default,
    allow,
    ask,
    explicit_overrides: [...draft.overrides],
    capabilities_required: persona?.tool_policy?.capabilities_required || [],
  }
}

function sameSet(a: Set<string>, b: Set<string>): boolean {
  if (a.size !== b.size) return false
  for (const x of a) if (!b.has(x)) return false
  return true
}

function policyChanged(a: PolicyDraft, b: PolicyDraft): boolean {
  if (a.default !== b.default) return true
  if (!sameSet(a.overrides, b.overrides)) return true
  const keys = new Set([...Object.keys(a.states), ...Object.keys(b.states)])
  for (const k of keys) if (a.states[k] !== b.states[k]) return true
  return false
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
  const [note, setNote] = useState<{ kind: 'ok' | 'warn' | 'err'; text: string } | null>(null)
  const [advOpen, setAdvOpen] = useState(false)
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [draft, setDraft] = useState<PolicyDraft>(() => deriveDraft(persona, tools))

  // Group the catalog by owning service so the (often long) per-service tool
  // sets can be collapsed. Null binding = built-in/local; sorted last.
  const groups = useMemo(() => {
    const by = new Map<string, ToolDef[]>()
    for (const t of tools) {
      const key = t.service_binding || BUILTIN_GROUP
      const arr = by.get(key)
      if (arr) arr.push(t)
      else by.set(key, [t])
    }
    return [...by.entries()].sort(([a], [b]) => {
      if (a === BUILTIN_GROUP) return 1
      if (b === BUILTIN_GROUP) return -1
      return a.localeCompare(b)
    })
  }, [tools])

  // distinct services present in the catalog — the bindings that can be granted
  const knownBindings = useMemo(
    () => [...new Set(tools.map((t) => t.service_binding).filter((b): b is string => !!b))].sort(),
    [tools],
  )

  // Re-derive the baseline whenever the canonical persona changes (e.g. after a
  // save → refetch); dirtiness is then measured against the saved values.
  const baseline = useMemo(() => deriveDraft(persona, tools), [persona, tools])
  const policyDirty = useMemo(() => policyChanged(draft, baseline), [draft, baseline])
  const bindingsDirty = useMemo(
    () => !sameSet(draft.bindings, baseline.bindings),
    [draft, baseline],
  )
  const dirty = policyDirty || bindingsDirty

  const setToolState = (name: string, s: ToolState) =>
    setDraft((d) => ({ ...d, states: { ...d.states, [name]: s } }))
  const setDefault = (def: string) => setDraft((d) => ({ ...d, default: def }))
  const toggleBinding = (b: string) =>
    setDraft((d) => {
      const next = new Set(d.bindings)
      if (next.has(b)) next.delete(b)
      else next.add(b)
      return { ...d, bindings: next }
    })
  const toggleOverride = (key: string) =>
    setDraft((d) => {
      const next = new Set(d.overrides)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return { ...d, overrides: next }
    })
  const toggleGroup = (key: string) =>
    setCollapsed((c) => {
      const next = new Set(c)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  const reset = () => {
    setDraft(deriveDraft(persona, tools))
    setNote(null)
  }

  // one tool's row (tri-state + capability badges)
  const toolRow = (t: ToolDef) => {
    const caps = t.capabilities
    const st = draft.states[t.name] ?? 'off'
    return (
      <div className="toolrow" key={t.name}>
        <div className="tt">
          <span className="nm">{t.name}</span>
          <div className="tristate" role="group" aria-label={`${t.name} policy`}>
            {(['off', 'allow', 'ask'] as ToolState[]).map((s) => (
              <button
                key={s}
                className={'tri' + (st === s ? ' on ' + s : '')}
                disabled={busy}
                title={
                  s === 'off'
                    ? 'tool hidden from the model'
                    : s === 'allow'
                      ? 'auto-runs when the model calls it'
                      : 'parks for CONFIRM approval before running'
                }
                onClick={() => setToolState(t.name, s)}
              >
                {s}
              </button>
            ))}
          </div>
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
  }

  const onSave = async () => {
    if (!persona) return
    setBusy(true)
    setNote(null)
    try {
      // service_bindings first (gate 2 — controls which tools are even
      // offered), then tool_policy. Both go through dev_command, which
      // persists to personas.json on mutation. Send only what changed.
      let last = { response: '', mutated: false } as { response: string; mutated?: boolean }
      if (bindingsDirty) {
        const list = [...draft.bindings].join(',') || 'none'
        last = await store.runToolsCommand(`set service_bindings ${list}`)
      }
      if (policyDirty) {
        const policy = draftToPolicy(draft, tools, persona)
        // compact JSON (no spaces) survives the dev-command whitespace tokenizer
        last = await store.runToolsCommand(`set tool_policy ${JSON.stringify(policy)}`)
      }
      if (last.mutated) {
        // a now-quarantined persona comes back in the response text + the banner
        const warn = /quarantin|insecure/i.test(last.response || '')
        setNote({ kind: warn ? 'warn' : 'ok', text: last.response || 'saved' })
      } else {
        setNote({ kind: 'err', text: last.response || 'rejected — unchanged' })
      }
    } catch (e) {
      setNote({ kind: 'err', text: `save failed: ${e}` })
    } finally {
      setBusy(false)
    }
  }

  if (!persona) return <div className="dimrow">no persona loaded</div>

  return (
    <div className="insp-pane active">
      {persona.security_blocked && (
        <div className="secbanner" style={{ display: 'block' }}>
          ⚠ security-blocked — generation refused until tools are scoped to a safe set
        </div>
      )}
      {note && <div className={'toolnote ' + note.kind}>{note.text}</div>}

      <div className="polhead">
        <span className="lbl">default</span>
        <select value={draft.default} onChange={(e) => setDefault(e.target.value)}>
          <option value="deny">deny — only listed tools</option>
          <option value="allow">allow — all tools (incl. wildcard)</option>
        </select>
      </div>

      {knownBindings.length > 0 && (
        <>
          <div className="section">
            service bindings
            <span className="desc">a service&apos;s tools are inert until its binding is granted</span>
          </div>
          <div className="bindbody">
            {knownBindings.map((b) => (
              <label className="bindrow" key={b}>
                <button
                  type="button"
                  className={'toggle-chip' + (draft.bindings.has(b) ? ' on' : '')}
                  disabled={busy}
                  onClick={() => toggleBinding(b)}
                >
                  <span className="sw" />
                </button>
                <span className="bindlbl">{b}</span>
                {!draft.bindings.has(b) && <span className="bindoff">tools hidden</span>}
              </label>
            ))}
          </div>
        </>
      )}

      {groups.map(([key, groupTools]) => {
        const bindingOff = key !== BUILTIN_GROUP && !draft.bindings.has(key)
        const isCollapsed = collapsed.has(key)
        const nAllow = groupTools.filter((t) => draft.states[t.name] === 'allow').length
        const nAsk = groupTools.filter((t) => draft.states[t.name] === 'ask').length
        const label = key === BUILTIN_GROUP ? 'built-in · local' : key
        return (
          <div key={key}>
            <div
              className={'section toolgrp' + (isCollapsed ? ' collapsed' : '')}
              onClick={() => toggleGroup(key)}
            >
              {label}
              <span className="pill">{groupTools.length}</span>
              {bindingOff && <span className="pill bindwarn">binding off</span>}
              <span className="desc">
                {nAllow ? `${nAllow} allow` : ''}
                {nAllow && nAsk ? ' · ' : ''}
                {nAsk ? `${nAsk} ask` : ''}
                {!nAllow && !nAsk ? 'none enabled' : ''}
              </span>
              <span className="car">▾</span>
            </div>
            {!isCollapsed && groupTools.map((t) => toolRow(t))}
          </div>
        )
      })}

      <div
        className={'section adv' + (advOpen ? '' : ' collapsed')}
        onClick={() => setAdvOpen((o) => !o)}
      >
        ⚠ security overrides
        <span className="desc">disable a composition guard — only if you know why</span>
        <span className="car">▾</span>
      </div>
      {advOpen && (
        <div className="advbody">
          {OVERRIDE_FLAGS.map((f) => (
            <label className="ovrow" key={f.key}>
              <input
                type="checkbox"
                checked={draft.overrides.has(f.key)}
                disabled={busy}
                onChange={() => toggleOverride(f.key)}
              />
              <span className="ovlbl">
                {f.label} <code>{f.key}</code>
              </span>
            </label>
          ))}
        </div>
      )}

      <div className="savebar">
        {note && <span className={'savestatus ' + note.kind}>{note.text}</span>}
        <span className="grow" />
        <button className="mini" onClick={reset} disabled={!dirty || busy}>
          reset
        </button>
        <button className="savebtn" onClick={onSave} disabled={!dirty || busy}>
          {busy ? 'saving…' : 'save'}
        </button>
      </div>
    </div>
  )
}

// Raw req — THE parity inspector (S5/DP-137). Renders the exact request the
// engine would send, sourced from GET /assemble (the shared live builder), so
// the parity banner is green ONLY when source === 'engine.dry_run'. A
// pane-local preview field lets you dry-run a hypothetical next user turn
// without sending it; empty = the request as it stands.
function RawPane({ store }: { store: PortalStore }) {
  const { persona, assembled, fetchAssembled, offline } = store
  const [preview, setPreview] = useState('')

  // Re-fetch when the active persona changes (keyed on persona name).
  useEffect(() => {
    fetchAssembled('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [persona?.name])

  const verified = assembled?.parity.source === 'engine.dry_run'

  return (
    <div className="insp-pane raw active">
      {/* parity banner — green when the engine's shared builder produced it,
          red on client fallback (engine unreachable → mock). */}
      {assembled ? (
        <div className={'parity ' + (verified ? 'ok' : 'warn')}>
          <span className={'pi' + (verified ? '' : ' warn')}>
            {verified
              ? '✓ parity verified — dry-run of the live builder'
              : '⚠ client fallback — may drift'}
          </span>
          <span className="pd">
            {verified ? (
              <>
                Same code path as a live submit: <code>{assembled.parity.builder}</code>{' '}
                via the shared <code>_prepare_request</code> +{' '}
                <code>build_wire_messages</code> helpers — not reconstructed
                client-side.
              </>
            ) : (
              <>
                The engine&apos;s <code>/assemble</code> dry-run is unavailable
                {offline ? ' (offline · mock)' : ''}, so this is a client
                approximation and is <b>not</b> guaranteed to match what a live
                submit assembles.
              </>
            )}
          </span>
        </div>
      ) : (
        <div className="rawfoot">loading assembled request…</div>
      )}

      {/* preview a hypothetical next user turn (does NOT send it) */}
      <div className="rawsec">
        <span className="rk">preview turn</span>
        <span className="rv">
          <input
            value={preview}
            onChange={(e) => setPreview(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') fetchAssembled(preview)
            }}
            placeholder="(optional) dry-run a new user message"
            style={{
              width: '100%',
              background: 'var(--panel-3)',
              border: '1px solid var(--line-2)',
              borderRadius: 5,
              color: 'var(--ink)',
              font: 'inherit',
              fontSize: 11,
              padding: '4px 6px',
            }}
          />
          <button
            className="mini"
            style={{ marginTop: 6 }}
            onClick={() => fetchAssembled(preview)}
          >
            re-assemble
          </button>
        </span>
      </div>

      {assembled && (
        <>
          {/* routing */}
          <div className="rawsec">
            <span className="rk">route</span>
            <span className="rv accent">{assembled.route}</span>
            <span className="rk">model_name</span>
            <span className="rv">{assembled.model_name}</span>
          </div>

          {/* local_inference_config — resolved sampling params forwarded */}
          <div className="rawlbl">local_inference_config</div>
          <div className="rawsec">
            {Object.entries(assembled.params).map(([k, v]) => (
              <RawKV key={k} k={k} v={v} />
            ))}
          </div>

          {/* messages[] — each wire line tagged to its source row */}
          <div className="rawlbl wrap">
            messages[{assembled.messages.length}]
            <span className="rawnote">
              History is rebuilt from the DB; the client message array is
              discarded. Each line maps back to its source row.
            </span>
          </div>
          {assembled.messages.map((m, i) => (
            <div key={i} className={'wire ' + m.role}>
              <div className="wh">
                <span className="wrole">⟦{m.role}⟧</span>
                <span className="wsrc">{m.src}</span>
                <span className="widx">{fmtTok(estimateTokens(m.content))} tok</span>
              </div>
              <div className="wtext">{m.content}</div>
            </div>
          ))}

          <div className="rawfoot">
            Edit any line by editing its source row — never a free-text blob. The
            next submit re-runs this exact assembly.
          </div>
        </>
      )}
    </div>
  )
}

// One key/value row in the local_inference_config block. null/undefined render
// dimmed-italic; everything else stringified.
function RawKV({ k, v }: { k: string; v: unknown }) {
  const isNull = v === null || v === undefined
  return (
    <>
      <span className="rk">{k}</span>
      <span className={'rv' + (isNull ? ' null' : '')}>
        {isNull ? 'null' : typeof v === 'object' ? JSON.stringify(v) : String(v)}
      </span>
    </>
  )
}
