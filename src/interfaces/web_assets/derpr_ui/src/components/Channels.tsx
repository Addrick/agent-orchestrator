import { useState } from 'react'
import type { ChannelGroup } from '../types/contracts'

interface Props {
  channels: ChannelGroup[]
}

export function Channels({ channels }: Props) {
  const [activeId, setActiveId] = useState<string | null>(
    channels.flatMap((g) => g.items).find((i) => i.active)?.id ?? null,
  )

  return (
    <div className="col chan">
      <div className="chead">
        <span>Channels</span>
        <span className="grow" />
        <button className="ibtn" title="new channel">
          +
        </button>
      </div>
      <div className="searchwrap">
        <div className="search">⌕ filter channels…</div>
      </div>
      <div className="chanlist">
        {channels.map((g) => (
          <div key={g.group}>
            <div className="changroup">{g.group}</div>
            {g.items.map((it) => (
              <button
                key={it.id}
                className={'chanitem' + (it.id === activeId ? ' active' : '')}
                onClick={() => setActiveId(it.id)}
              >
                <div className="av">{it.name.slice(0, 2).toUpperCase()}</div>
                <div className="ci">
                  <div className="top">
                    <span className="nm">{it.name}</span>
                    <span className={'src ' + it.source}>{it.source}</span>
                  </div>
                  <div className="pv">{it.preview}</div>
                </div>
              </button>
            ))}
          </div>
        ))}
        <button className="newchan">+ new web_ui channel</button>
      </div>
    </div>
  )
}
