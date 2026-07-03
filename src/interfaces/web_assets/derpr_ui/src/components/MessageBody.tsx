import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface Props {
  content: string
}

// Streaming-safe: an unterminated $...$/$$...$$ mid-token just renders as
// literal text until the closing delimiter arrives, no crash.
export function MessageBody({ content }: Props) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
      {content}
    </ReactMarkdown>
  )
}
