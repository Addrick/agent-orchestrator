import { useState } from 'react'
import { estimateTokens, fmtTok } from '../state/util'

export function ReasoningFold({ reasoning: raw }: { reasoning: string }) {
  const [open, setOpen] = useState(false)
  // A later step's thinking streams in after a paragraph break (DP-434).
  const reasoning = raw.trim()
  return (
    <div className={'fold' + (open ? ' open' : '')}>
      <button className="fh" onClick={() => setOpen((o) => !o)}>
        <span className="tw">⟁ reasoning</span>
        <span style={{ color: 'var(--ink-faint)' }}>{fmtTok(estimateTokens(reasoning))} tok</span>
        <span className="car">▸</span>
      </button>
      <div className="fb">
        <div className="t">{reasoning}</div>
      </div>
    </div>
  )
}
