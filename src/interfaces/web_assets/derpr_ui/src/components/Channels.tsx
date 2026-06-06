import { useState } from 'react'
import type { PortalStore } from '../state/store'

interface Props {
  store: PortalStore
}

/** Channel rail (DP-136 6b). Lists the persona's channels grouped by source,
 *  switches the active channel (re-scopes the transcript), and creates a new
 *  web_ui channel (materializes on the first submit). The active channel is
 *  driven by `store.activeChannel`, not local position. */
export function Channels({ store }: Props) {
  const { channels, activeChannel, switchChannel, newChannel } = store
  const [filter, setFilter] = useState('')

  const onNew = () => {
    const name = window.prompt('New channel name (web_ui prefix added if absent):')
    if (name === null) return
    void newChannel(name)
  }

  const groups = channels
    .map((g) => ({
      ...g,
      items: g.items.filter(
        (it) =>
          !filter ||
          it.name.toLowerCase().includes(filter.toLowerCase()) ||
          it.channel.toLowerCase().includes(filter.toLowerCase()),
      ),
    }))
    .filter((g) => g.items.length)

  return (
    <div className="col chan">
      <div className="chead">
        <span>Channels</span>
        <span className="grow" />
        <button className="ibtn" title="new channel" onClick={onNew}>
          +
        </button>
      </div>
      <div className="searchwrap">
        <input
          className="search"
          placeholder="⌕ filter channels…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      <div className="chanlist">
        {groups.map((g) => (
          <div key={g.group}>
            <div className="changroup">{g.group}</div>
            {g.items.map((it) => (
              <button
                key={it.id}
                className={
                  'chanitem' + (it.channel === activeChannel ? ' active' : '')
                }
                onClick={() => void switchChannel(it.channel)}
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
        <button className="newchan" onClick={onNew}>
          + new web_ui channel
        </button>
      </div>
    </div>
  )
}
