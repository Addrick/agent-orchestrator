// The CHAT dock is the only built destination. The five expansion docks are
// rendered disabled with a "soon" dot per the spec — DO NOT build them.
const DOCKS = [
  { ic: '▤', t: 'CHAT', active: true, tip: 'Chat workspace (Control Room)' },
  { ic: '◈', t: 'MEMORY', soon: true, tip: 'Memory inspector — soon' },
  { ic: '⊞', t: 'AGENTS', soon: true, tip: 'Agent monitor — soon' },
  { ic: '▰', t: 'BUDGET', soon: true, tip: 'Budget visualizer — soon' },
  { ic: '◔', t: 'STATS', soon: true, tip: 'Analytics / cost — soon' },
  { ic: '❏', t: 'PERSONA', soon: true, tip: 'Persona library — soon' },
]

export function NavRail() {
  return (
    <div className="col rail">
      <div className="rail" style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        {DOCKS.map((d) => (
          <button
            key={d.t}
            className={'railitem' + (d.active ? ' active' : '') + (d.soon ? ' soon' : '')}
            data-tip={d.tip}
            disabled={d.soon}
            aria-disabled={d.soon ? 'true' : 'false'}
          >
            {d.soon && <span className="badge-soon" />}
            <span className="ic">{d.ic}</span>
            <span className="t">{d.t}</span>
          </button>
        ))}
        <div className="grow" />
        <button className="railitem" data-tip="settings" title="settings">
          <span className="ic">⚙</span>
          <span className="t">CFG</span>
        </button>
      </div>
    </div>
  )
}
