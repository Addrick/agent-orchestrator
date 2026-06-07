import type { Chunk, Persona } from '../types/contracts'
import { splitThink, estimateTokens, fmtTok, argString } from '../state/util'

interface Props {
  persona: Persona
  chunks: Chunk[]
  ltmBlock: string | null
  ltmOn: boolean
}

// CONTEXT ↦ LLM view: the assembled prompt as role-tagged rows, with per-row
// token counts. System + author's-note are stitched (not transcript chunks).
// NOTE: this is a client-side approximation. The authoritative, parity-verified
// assembled request comes from the /assemble dry-run endpoint (S5, Raw req tab).
export function ContextView({ persona, chunks, ltmBlock, ltmOn }: Props) {
  const rows: { cls: string; role: string; text: string }[] = []
  rows.push({ cls: 'system', role: "⟦system⟧", text: persona.prompt })
  if (ltmOn && ltmBlock)
    rows.push({ cls: 'anote', role: "⟦author's-note · LTM⟧", text: ltmBlock })

  // visible turns only — skip ephemeral parked confirmation (not yet in prompt)
  chunks
    .filter((c) => !c.ephemeral)
    .forEach((c) => {
      const { body } = splitThink(c.content)
      const tools = (c.tool_context || [])
        .map(
          (t) =>
            `  → ${t.tool_name}(${argString(t.arguments)}) ⇒ ${t.result || '—'}`,
        )
        .join('\n')
      const text = tools ? (body ? body + '\n' + tools : tools) : body
      rows.push({ cls: c.role, role: `⟦${c.role}⟧`, text })
    })

  return (
    <>
      <div className="ctxnote">
        <b>Approximate assembled view</b> — client-reconstructed. The
        parity-verified request (same code path as a live submit) is the Raw req
        inspector tab via <b>/assemble</b> (S5). System + author's-note are not
        transcript chunks; they're stitched here.
      </div>
      {rows.map((r, i) => (
        <div key={i} className={'ctxrow ' + r.cls}>
          <div className="lh">
            <span className="role">{r.role}</span>
            <span className="tcount">{fmtTok(estimateTokens(r.text))} tok</span>
          </div>
          <div className="text">{r.text}</div>
        </div>
      ))}
    </>
  )
}
