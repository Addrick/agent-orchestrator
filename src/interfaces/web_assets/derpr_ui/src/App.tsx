import { useEffect, useState } from 'react'
import { usePortalStore } from './state/store'
import { readPref, writePref } from './state/persist'
import { TopBar } from './components/TopBar'
import { NavRail } from './components/NavRail'
import { Channels } from './components/Channels'
import { Conversation } from './components/Conversation'
import { Inspector } from './components/Inspector'
import { NewPersonaModal } from './components/NewPersonaModal'

interface Collapsed {
  rail: boolean
  chan: boolean
  insp: boolean
}

const COLLAPSED_DEFAULT: Collapsed = { rail: false, chan: false, insp: false }

// Coerce a persisted collapse blob to Collapsed — any missing/garbage field
// falls back to expanded (DP-273).
function validateCollapsed(raw: unknown): Collapsed {
  const o = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  return { rail: Boolean(o.rail), chan: Boolean(o.chan), insp: Boolean(o.insp) }
}

export default function App() {
  const store = usePortalStore()
  const [collapsed, setCollapsed] = useState<Collapsed>(() =>
    readPref('collapsed', COLLAPSED_DEFAULT, validateCollapsed),
  )
  const [newPersonaOpen, setNewPersonaOpen] = useState(false)

  // The grid collapse rules in theme.css key off classes on <body>. Persist the
  // panel folds so the layout survives a reload (DP-273).
  useEffect(() => {
    const b = document.body
    b.classList.toggle('no-rail', collapsed.rail)
    b.classList.toggle('no-chan', collapsed.chan)
    b.classList.toggle('no-insp', collapsed.insp)
    writePref('collapsed', collapsed)
  }, [collapsed])

  const toggle = (k: keyof Collapsed) =>
    setCollapsed((c) => ({ ...c, [k]: !c[k] }))

  return (
    <div className="app">
      <TopBar
        store={store}
        collapsed={collapsed}
        toggle={toggle}
        onNewPersona={() => setNewPersonaOpen(true)}
      />
      <div className="body">
        <NavRail />
        <Channels store={store} />
        <Conversation store={store} />
        <Inspector store={store} />
      </div>
      {newPersonaOpen && (
        <NewPersonaModal store={store} onClose={() => setNewPersonaOpen(false)} />
      )}
    </div>
  )
}
