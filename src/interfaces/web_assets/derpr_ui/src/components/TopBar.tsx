import type { PortalStore } from '../state/store'

interface Props {
  store: PortalStore
  collapsed: { rail: boolean; chan: boolean; insp: boolean }
  toggle: (k: 'rail' | 'chan' | 'insp') => void
}

export function TopBar({ store, collapsed, toggle }: Props) {
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
