import type { Persona } from '../types/contracts'
import { estimateTokens, fmtTok } from '../state/util'

interface Props {
  persona: Persona | null
  systemPrompt: string
  ltmBlock: string | null
  ltmOn: boolean
  historyText: string
}

// Budget bar. Token numbers should come from POST /api/extra/tokencount or the
// /assemble payload in production; here we estimate client-side (DEMO ONLY) as
// the contract endpoints aren't wired until S4/S5. Marked accordingly.
export function BudgetBar({ persona, systemPrompt, ltmBlock, ltmOn, historyText }: Props) {
  const max = persona?.max_context_tokens ?? 16384
  const reserve = persona?.max_tokens ?? 1024

  const sysTok = estimateTokens(systemPrompt)
  const ltmTok = ltmOn ? estimateTokens(ltmBlock || '') : 0
  const histTok = estimateTokens(historyText)

  const segs = [
    { label: 'system prompt', tokens: sysTok, color: 'var(--accent-dim)' },
    ...(ltmOn ? [{ label: 'LTM / anote', tokens: ltmTok, color: 'var(--mem)' }] : []),
    { label: 'history', tokens: histTok, color: 'rgba(150,170,205,0.45)' },
    { label: 'reply reserve', tokens: reserve, color: 'var(--write)' },
  ]
  const used = segs.reduce((a, s) => a + s.tokens, 0)

  return (
    <div className="budget" id="budget">
      <div className="bar">
        {segs.map((s, i) => (
          <i
            key={i}
            style={{ background: s.color, width: (s.tokens / max) * 100 + '%' }}
          />
        ))}
      </div>
      <div className="legend">
        {segs.map((s, i) => (
          <span key={i}>
            <i style={{ background: s.color }} />
            {s.label} {fmtTok(s.tokens)}
          </span>
        ))}
        <span className="total" title="client estimate — tokencount/assemble wires it in S4/S5">
          ~{fmtTok(used)} / {fmtTok(max)} ctx
        </span>
      </div>
    </div>
  )
}
