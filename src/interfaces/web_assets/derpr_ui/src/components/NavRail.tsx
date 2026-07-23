import type { PortalView } from '../App'

// CHAT + MEMORY are built (MEMORY = DP-292 import panel). The remaining docks
// stay disabled with a "soon" dot per the spec — DO NOT build them.
interface Dock {
  ic: string
  t: string
  view?: PortalView
  soon?: boolean
  tip: string
}

const DOCKS: Dock[] = [
  { ic: '▤', t: 'CHAT', view: 'chat', tip: 'Chat workspace (Control Room)' },
  { ic: '◈', t: 'MEMORY', view: 'memory', tip: 'Memory import panel' },
  { ic: '⊞', t: 'AGENTS', soon: true, tip: 'Agent monitor — soon' },
  { ic: '▰', t: 'BUDGET', soon: true, tip: 'Budget visualizer — soon' },
  { ic: '◔', t: 'STATS', soon: true, tip: 'Analytics / cost — soon' },
  { ic: '❏', t: 'PERSONA', soon: true, tip: 'Persona library — soon' },
]

export function NavRail({
  view,
  onNavigate,
}: {
  view: PortalView
  onNavigate: (v: PortalView) => void
}) {
  return (
    <div className="col rail">
      <div className="rail" style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        {DOCKS.map((d) => {
          const active = d.view != null && d.view === view
          return (
            <button
              key={d.t}
              className={'railitem' + (active ? ' active' : '') + (d.soon ? ' soon' : '')}
              data-tip={d.tip}
              disabled={d.soon}
              aria-disabled={d.soon ? 'true' : 'false'}
              onClick={() => d.view && onNavigate(d.view)}
            >
              {d.soon && <span className="badge-soon" />}
              <span className="ic">{d.ic}</span>
              <span className="t">{d.t}</span>
            </button>
          )
        })}
        <div className="grow" />
        <button className="railitem" data-tip="settings" title="settings">
          <span className="ic">⚙</span>
          <span className="t">CFG</span>
        </button>
      </div>
    </div>
  )
}
