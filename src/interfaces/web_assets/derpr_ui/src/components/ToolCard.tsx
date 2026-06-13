import { useState } from 'react'
import type { ToolContext, ToolDef } from '../types/contracts'
import { argString } from '../state/util'

interface Props {
  tc: ToolContext
  tools: ToolDef[]
}

export function ToolCard({ tc, tools }: Props) {
  const [open, setOpen] = useState(false)
  const def = tools.find((t) => t.name === tc.tool_name)
  const isWrite = def?.is_write ?? false
  const caps = def?.capabilities

  const badges: React.ReactNode[] = [
    isWrite ? (
      <span key="w" className="badge write">
        write
      </span>
    ) : (
      <span key="r" className="badge read">
        read
      </span>
    ),
  ]
  if (caps?.sensitivity === 'high')
    badges.push(
      <span key="s" className="badge high">
        sensitive
      </span>,
    )
  else if (caps?.sensitivity === 'medium')
    badges.push(
      <span key="s" className="badge med">
        medium
      </span>,
    )
  if (caps?.locality)
    badges.push(
      <span key="l" className={'badge ' + caps.locality}>
        {caps.locality}
      </span>,
    )
  if (caps?.produces_untrusted)
    badges.push(
      <span key="u" className="badge low">
        untrusted out
      </span>,
    )

  return (
    <div className={'tool' + (isWrite ? '' : ' read') + (open ? ' open' : '')}>
      <button className="th" onClick={() => setOpen((o) => !o)}>
        <span className="fn">
          → <b>{tc.tool_name}</b>({argString(tc.arguments)})
        </span>
        {badges}
        <span className="car">▸</span>
      </button>
      <div className="tb">
        <div className="kv">
          <span className="k">call_id</span>
          <span className="v">
            {tc.call_id}
            {tc.group_id ? ' · ' + tc.group_id : ''}
          </span>
        </div>
        <div className="kv">
          <span className="k">args</span>
          <span className="v">{JSON.stringify(tc.arguments)}</span>
        </div>
        {tc.result != null ? (
          <div className="result ok">{tc.result}</div>
        ) : tc.error ? (
          <div className="result" style={{ color: 'var(--danger)' }}>
            {tc.error}
          </div>
        ) : (
          <div className="result pending">awaiting approval — tool not yet run</div>
        )}
      </div>
    </div>
  )
}
