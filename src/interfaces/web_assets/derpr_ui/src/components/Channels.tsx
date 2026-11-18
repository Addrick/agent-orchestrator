import { useState } from 'react'
import type { PortalStore } from '../state/store'
import { fmtWhen, fmtWhenFull } from '../state/util'

interface Props {
  store: PortalStore
}

/** Channel rail (DP-136 6b). Lists the persona's channels grouped by source,
 *  switches the active channel (re-scopes the transcript), and creates a new
 *  web_ui channel (materializes on the first submit). The active channel is
 *  driven by `store.activeChannel`, not local position.
 *
 *  DP-523: under a merged memory mode every channel shows the same
 *  conversation, so the rail says so and offers no switch or new channel. */
export function Channels({ store }: Props) {
  const { channels, activeChannel, channelsMerged, switchChannel, newChannel, persona } = store
  const [filter, setFilter] = useState('')

  const onNew = () => {
    const name = window.prompt('New channel name (web_ui prefix added if absent):')
    if (name === null) return
    void newChannel(name).catch((err) => {
      console.error('Failed to create channel:', err)
    })
  }

  const groups = channels
    .map((g) => ({
      ...g,
      items: g.items.filter(
        (it) => !filter || it.label.toLowerCase().includes(filter.toLowerCase()),
      ),
    }))
    .filter((g) => g.items.length)

  return (
    <div className="col chan">
      <div className="chead">
        <span>Channels</span>
        <span className="grow" />
        {!channelsMerged && (
          <button className="ibtn" title="new channel" onClick={onNew}>
            +
          </button>
        )}
      </div>
      {channelsMerged && (
        <div className="chanmerged">
          {persona?.memory_mode} memory: every channel shows one merged conversation
        </div>
      )}
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
                disabled={channelsMerged}
                onClick={() => void switchChannel(it.channel)}
              >
                <div className="av">{it.label.slice(0, 2).toUpperCase()}</div>
                <div className="ci">
                  <div className="top">
                    <span className="nm" title={it.channel}>{it.label}</span>
                    <span className={'src ' + it.source}>{it.source}</span>
                  </div>
                  <div className="pv">
                    {it.preview}
                    {it.lastTs && (
                      <span title={'last message ' + fmtWhenFull(it.lastTs)}>
                        {' · '}
                        {fmtWhen(it.lastTs)}
                      </span>
                    )}
                  </div>
                </div>
              </button>
            ))}
          </div>
        ))}
        {!channelsMerged && (
          <button className="newchan" onClick={onNew}>
            + new web_ui channel
          </button>
        )}
      </div>
    </div>
  )
}
