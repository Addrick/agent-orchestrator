import { useState } from 'react'
import { getControlToken, setControlToken, hasControlToken } from '../api/control_token'
import type { PortalStore } from '../state/store'

interface Props {
  store: PortalStore
  collapsed: { rail: boolean; chan: boolean; insp: boolean }
  toggle: (k: 'rail' | 'chan' | 'insp') => void
  onNewPersona: () => void
}

// DP-277: operator token control. The token gates every control-plane route
// server-side; entering it here (kept in localStorage, never shown to the
// model) is what lets this browser create/edit personas and run dev commands.
function OperatorToken() {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(getControlToken())
  const active = hasControlToken()
  if (!editing) {
    return (
      <button
        className="stat"
        title={active ? 'operator token set — click to change' : 'set operator token to enable persona/config edits'}
        onClick={() => { setValue(getControlToken()); setEditing(true) }}
      >
        <span className="dot" style={active ? undefined : { background: 'var(--write)', boxShadow: '0 0 7px var(--write)' }} />
        {active ? 'operator ✓' : 'operator: locked'}
      </button>
    )
  }
  return (
    <span className="stat" style={{ gap: 4 }}>
      <input
        type="password"
        autoFocus
        placeholder="control token"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') { setControlToken(value); setEditing(false) }
          if (e.key === 'Escape') setEditing(false)
        }}
        style={{ width: 120, background: 'transparent', border: '1px solid var(--line)', color: 'inherit', font: 'inherit', padding: '1px 4px' }}
      />
      <button className="cbtn" title="save token" onClick={() => { setControlToken(value); setEditing(false) }}>✓</button>
      <button className="cbtn" title="clear token" onClick={() => { setControlToken(''); setValue(''); setEditing(false) }}>✕</button>
    </span>
  )
}

export function TopBar({ store, collapsed, toggle, onNewPersona }: Props) {
  const { persona, personaList, activePersona, switchPersona, offline } = store
  const initials = (persona?.display_name || activePersona || 'AS').slice(0, 2).toUpperCase()

  return (
    <div className="topbar">
      <div className="brand">
        <b>DERPR</b>
        <span className="v">// PORTAL · ENGINE</span>
      </div>

      <label className="persona-pick" title="active persona (PUT /api/v1/model)">
        <span className="av">{initials}</span>
        <select
          className="nm"
          value={activePersona}
          onChange={(e) => switchPersona(e.target.value)}
        >
          {personaList.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <span className="car">▾</span>
      </label>

      <button className="newpersona-btn" title="create a new persona" onClick={onNewPersona}>
        + new
      </button>

      <div className="spacer" />

      <div className="statgrp">
        <span className="stat" title="connection">
          <span className="dot" style={offline ? { background: 'var(--write)', boxShadow: '0 0 7px var(--write)' } : undefined} />
          {offline ? 'mock · offline' : 'kcpp · engine:5003'}
        </span>
        <span className="stat route">
          <span className="dot" />
          engine route · /v1/chat/completions
        </span>
        <OperatorToken />
      </div>

      <div className="collapse-btns">
        <button
          className="cbtn"
          id="tg-rail"
          aria-pressed={!collapsed.rail}
          title="toggle nav rail"
          onClick={() => toggle('rail')}
        >
          ⊟
        </button>
        <button
          className="cbtn"
          id="tg-chan"
          aria-pressed={!collapsed.chan}
          title="toggle channels"
          onClick={() => toggle('chan')}
        >
          ◧
        </button>
        <button
          className="cbtn"
          id="tg-insp"
          aria-pressed={!collapsed.insp}
          title="toggle inspector"
          onClick={() => toggle('insp')}
        >
          ◨
        </button>
      </div>
    </div>
  )
}
