import { useState } from 'react'
import type { Persona, ToolDef } from '../types/contracts'
import { policyLabel } from '../state/util'

type Tab = 'persona' | 'tools' | 'raw'

interface Props {
  persona: Persona | null
  tools: ToolDef[]
}

// Inspector chrome. The Persona tab renders the base-vs-kobold split read-only;
// edit/PATCH + rejection surfacing is S4, the parity /assemble Raw-req tab is S5
// — both out of scope here, so those panes show their intent without wiring.
export function Inspector({ persona, tools }: Props) {
  const [tab, setTab] = useState<Tab>('persona')

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
        {tab === 'persona' && <PersonaPane persona={persona} />}
        {tab === 'tools' && <ToolsPane persona={persona} tools={tools} />}
        {tab === 'raw' && <RawPane />}
      </div>
    </div>
  )
}

function PersonaPane({ persona: p }: { persona: Persona | null }) {
  const [koboldCollapsed, setKoboldCollapsed] = useState(true)
  if (!p) return <div className="dimrow">no persona loaded</div>
  const kx = p.kobold_extras || {}

  return (
    <div className="insp-pane active">
      {p.security_blocked && (
        <div className="secbanner" style={{ display: 'block' }}>
          ⚠ persona security-blocked —{' '}
          {(p.security_block_reasons || []).join('; ') || 'tooling disabled'}
        </div>
      )}
      <div className="field">
        <span className="lbl">Identity</span>
        <div className="ctrl">
          <span>{p.name}</span>
          <span style={{ color: 'var(--ink-faint)' }}>read-only · edit = S4</span>
        </div>
      </div>
      <div className="field">
        <span className="lbl">System prompt</span>
        <div className="ctrl area">{p.prompt}</div>
      </div>

      <div className="section">
        ▣ base params<span className="desc">sent to every provider</span>
      </div>
      <Row2 a={['model_name', p.model_name]} b={['memory_mode', p.memory_mode]} />
      <div className="field">
        <span className="lbl">
          temperature · {p.temperature != null ? p.temperature.toFixed(2) : '—'}
        </span>
        <div className="slider">
          <input type="range" min={0} max={2} step={0.05} value={p.temperature ?? 0} readOnly />
          <span className="val">{p.temperature != null ? p.temperature.toFixed(2) : '—'}</span>
        </div>
      </div>
      <Row2 a={['max_tokens', p.max_tokens]} b={['history_messages', p.history_messages]} />
      <Row2
        a={['max_context_tokens', p.max_context_tokens]}
        b={['thinking_level', p.thinking_level]}
      />
      <Row2 a={['chat_template', p.chat_template]} b={['tool_policy', policyLabel(p.tool_policy)]} />

      <div
        className={'section kobold' + (koboldCollapsed ? ' collapsed' : '')}
        onClick={() => setKoboldCollapsed((c) => !c)}
      >
        ⚠ kobold-only<span className="pill">passthrough route</span>
        <span className="desc">provider_extra · only on kcpp endpoint</span>
        <span className="car">▾</span>
      </div>
      <div className={'kobold-fields' + (koboldCollapsed ? ' collapsed' : '')}>
        <Row2 a={['top_p', p.top_p]} b={['top_k', p.top_k]} />
        <Row2 a={['rep_pen', kx.rep_pen]} b={['rep_pen_range', kx.rep_pen_range]} />
        <Row2 a={['min_p', kx.min_p]} b={['tfs', kx.tfs]} />
        <div className="field">
          <span className="lbl">mirostat · tau · eta</span>
          <div className="ctrl">
            <span>
              {kx.mirostat} · {kx.mirostat_tau} · {kx.mirostat_eta}
            </span>
          </div>
        </div>
        <div className="field">
          <span className="lbl">instruct_tags</span>
          <div className="ctrl">
            <span>
              {p.instruct_tags && Object.keys(p.instruct_tags).length ? 'ChatML (custom)' : '—'}
            </span>
          </div>
        </div>
        <div className="field">
          <span className="lbl">sampler_order</span>
          <div className="ctrl">
            <span>[{(kx.sampler_order || []).join(', ')}]</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function Row2({
  a,
  b,
}: {
  a: [string, unknown]
  b: [string, unknown]
}) {
  return (
    <div className="field">
      <div className="row2">
        <div>
          <span className="lbl">{a[0]}</span>
          <div className="ctrl">
            <span>{a[1] == null ? '—' : String(a[1])}</span>
          </div>
        </div>
        <div>
          <span className="lbl">{b[0]}</span>
          <div className="ctrl">
            <span>{b[1] == null ? '—' : String(b[1])}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function ToolsPane({ persona, tools }: { persona: Persona | null; tools: ToolDef[] }) {
  const enabled = new Set(persona?.enabled_tools || [])
  return (
    <div className="insp-pane active">
      {tools.map((t) => {
        const caps = t.capabilities
        const on = enabled.has(t.name)
        return (
          <div className="toolrow" key={t.name}>
            <div className="tt">
              <span className="nm">{t.name}</span>
              <span className={'en' + (on ? ' on' : '')} title={on ? 'enabled' : 'disabled'}>
                <span className="sw" />
              </span>
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
