import { useEffect, useState } from 'react'
import { usePortalStore } from './state/store'
import { TopBar } from './components/TopBar'
import { NavRail } from './components/NavRail'
import { Channels } from './components/Channels'
import { Conversation } from './components/Conversation'
import { Inspector } from './components/Inspector'

interface Collapsed {
  rail: boolean
  chan: boolean
  insp: boolean
}

export default function App() {
  const store = usePortalStore()
  const [collapsed, setCollapsed] = useState<Collapsed>({
    rail: false,
    chan: false,
    insp: false,
  })

  // The grid collapse rules in theme.css key off classes on <body>.
  useEffect(() => {
    const b = document.body
    b.classList.toggle('no-rail', collapsed.rail)
    b.classList.toggle('no-chan', collapsed.chan)
    b.classList.toggle('no-insp', collapsed.insp)
  }, [collapsed])

  const toggle = (k: keyof Collapsed) =>
    setCollapsed((c) => ({ ...c, [k]: !c[k] }))

  return (
    <div className="app">
      <TopBar store={store} collapsed={collapsed} toggle={toggle} />
      <div className="body">
        <NavRail />
        <Channels channels={store.channels} />
        <Conversation store={store} />
        <Inspector persona={store.persona} tools={store.tools} />
      </div>
    </div>
  )
}
