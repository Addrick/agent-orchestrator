import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface Props {
  content: string
}

type MdNode = { type: string; value?: string; children?: MdNode[] }

function splitSoftBreaks(node: MdNode): void {
  if (!node.children) return
  const out: MdNode[] = []
  for (const child of node.children) {
    if (child.type === 'text' && child.value && child.value.includes('\n')) {
      child.value.split('\n').forEach((part, i) => {
        if (i > 0) out.push({ type: 'break' })
        if (part) out.push({ type: 'text', value: part })
      })
    } else {
      splitSoftBreaks(child)
      out.push(child)
    }
  }
  node.children = out
}

// CommonMark collapses a lone "\n" inside a paragraph to a space; chat transcripts mean
// it as a real line break. Inheriting `white-space: pre-wrap` from `.msg .text` used to
// cover that, but it also rendered the "\n" separator mdast-util-to-hast puts between
// block siblings, which double-spaced the pane — so the `.md` wrapper opts out of
// pre-wrap and the breaks are promoted to nodes here instead. Same job as remark-breaks,
// without the dependency. Literal nodes (code, inlineCode, html, math) carry no text
// children, so recursion never reaches inside them.
function remarkSoftBreaks() {
  return (tree: unknown) => {
    splitSoftBreaks(tree as MdNode)
  }
}

// Streaming-safe: an unterminated $...$/$$...$$ mid-token just renders as
// literal text until the closing delimiter arrives, no crash.
export function MessageBody({ content }: Props) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkSoftBreaks]}
        rehypePlugins={[rehypeKatex]}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
