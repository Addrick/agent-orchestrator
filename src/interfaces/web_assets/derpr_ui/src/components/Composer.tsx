import { useState, useRef } from 'react'

interface Props {
  ltmOn: boolean
  onToggleLtm: () => void
  onSend: (text: string) => void
  onAbort: () => void
  streaming: boolean
}

export function Composer({ ltmOn, onToggleLtm, onSend, onAbort, streaming }: Props) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  const send = () => {
    const t = text.trim()
    if (!t || streaming) return
    onSend(t)
    setText('')
    if (ref.current) ref.current.style.height = 'auto'
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const grow = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }

  const isDev = text.trim().startsWith('/')

  return (
    <div className="composer">
      <div className="cbar">
        <button
          className={'toggle-chip' + (ltmOn ? ' on' : '')}
          onClick={onToggleLtm}
          title="fetch + inject ltm_block"
        >
          <span className="sw" />
          LTM recall
        </button>
        <span className="grow" />
        <span style={{ fontSize: 10, color: 'var(--ink-faint)' }}>
          {isDev ? 'dev-command → /dev_command' : 'leading / = dev-command'}
        </span>
        <span className="kbd">Enter</span>
        <span style={{ fontSize: 10, color: 'var(--ink-faint)' }}>send</span>
        <span className="kbd">⇧Enter</span>
        <span style={{ fontSize: 10, color: 'var(--ink-faint)' }}>newline</span>
      </div>
      <div className="cbox">
        <textarea
          ref={ref}
          value={text}
          onChange={grow}
          onKeyDown={onKeyDown}
          placeholder={streaming ? 'streaming…' : 'message the engine, or / for a dev command'}
          readOnly={streaming}
        />
        {streaming ? (
          <button className="send" style={{ borderColor: 'rgba(229,114,114,.4)', color: 'var(--danger)', background: 'var(--danger-bg)' }} onClick={onAbort}>
            ■ stop
          </button>
        ) : (
          <button className="send" onClick={send} disabled={!text.trim()}>
            SEND
          </button>
        )}
      </div>
    </div>
  )
}
