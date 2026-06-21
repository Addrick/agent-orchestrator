import { useState, useRef, useCallback, useEffect } from 'react'
import { useMicCapture } from './useMicCapture'
import { useMicStream } from './useMicStream'
import { transcribeVoice } from '../api/client'

interface Props {
  ltmOn: boolean
  onToggleLtm: () => void
  onSend: (text: string) => void
  onAbort: () => void
  streaming: boolean
}

// Auto-send preference persists across reloads: off by default (dictation goes
// into the draft so STT errors can be fixed), flip on once dictation is trusted.
const AUTOSEND_KEY = 'derpr.voice.autoSend'

export function Composer({ ltmOn, onToggleLtm, onSend, onAbort, streaming }: Props) {
  const [text, setText] = useState('')
  const [autoSend, setAutoSend] = useState(
    () => localStorage.getItem(AUTOSEND_KEY) === '1',
  )
  const [micStatus, setMicStatus] = useState<string | null>(null)
  const ref = useRef<HTMLTextAreaElement>(null)
  const mic = useMicCapture()

  const send = () => {
    const t = text.trim()
    if (!t || streaming) return
    onSend(t)
    setText('')
    if (ref.current) ref.current.style.height = 'auto'
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      // While streaming, let Enter insert a newline into the draft instead of
      // silently swallowing the keypress (send is gated anyway).
      if (streaming) return
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

  // Append dictated text to the draft and resize the textarea to fit.
  const appendDraft = (t: string) => {
    setText((prev) => (prev ? prev.replace(/\s+$/, '') + ' ' : '') + t)
    const el = ref.current
    if (el)
      requestAnimationFrame(() => {
        el.style.height = 'auto'
        el.style.height = Math.min(el.scrollHeight, 200) + 'px'
      })
  }

  // One rule for both mic modes: auto-send (when toggled on and the engine isn't
  // mid-stream) or drop into the draft to edit. Refs read by the streaming WS
  // callback so it never sees a stale toggle/streaming value.
  const autoSendRef = useRef(autoSend)
  const streamingRef = useRef(streaming)
  const onSendRef = useRef(onSend)
  useEffect(() => {
    autoSendRef.current = autoSend
    streamingRef.current = streaming
    onSendRef.current = onSend
  })

  const routeDictation = useCallback((t: string) => {
    if (autoSendRef.current && !streamingRef.current) onSendRef.current(t)
    else appendDraft(t)
    // appendDraft is stable enough for this hook's purpose; deps intentionally [].
  }, [])

  const stream = useMicStream(routeDictation)

  const setAutoSendPref = (on: boolean) => {
    setAutoSend(on)
    localStorage.setItem(AUTOSEND_KEY, on ? '1' : '0')
  }

  const micStart = async () => {
    if (mic.recording) return
    setMicStatus(null)
    const ok = await mic.start()
    if (!ok) setMicStatus('mic access denied')
  }

  const micStop = async () => {
    if (!mic.recording) return
    const chunk = await mic.stop()
    if (!chunk) {
      if (mic.error) setMicStatus(mic.error)
      return
    }
    setMicStatus('transcribing…')
    try {
      const t = (await transcribeVoice(chunk.pcm, chunk.sampleRate)).trim()
      if (!t) {
        setMicStatus("didn't catch that")
        return
      }
      setMicStatus(null)
      routeDictation(t)
    } catch {
      setMicStatus('transcribe failed')
    }
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
        {mic.supported && (
          <button
            className={'toggle-chip' + (autoSend ? ' on' : '')}
            onClick={() => setAutoSendPref(!autoSend)}
            title="auto-send dictation as a turn (off = drop into the draft to edit first)"
          >
            <span className="sw" />
            voice auto-send
          </button>
        )}
        {stream.supported && (
          <button
            className={'toggle-chip' + (stream.active ? ' on' : '')}
            onClick={stream.toggle}
            title="always-listening dictation — continuously streams the mic and drops what you say into the composer"
          >
            <span className="sw" />
            {stream.active ? 'listening…' : 'listen'}
          </button>
        )}
        <span className="grow" />
        {stream.error && (
          <span style={{ fontSize: 10, color: 'var(--danger)' }}>{stream.error}</span>
        )}
        {micStatus && (
          <span style={{ fontSize: 10, color: 'var(--ink-faint)' }}>{micStatus}</span>
        )}
        <span style={{ fontSize: 10, color: 'var(--ink-faint)' }}>
          {isDev ? 'dev-command → /dev_command' : 'leading / = dev-command'}
        </span>
        <span className="kbd">Enter</span>
        <span style={{ fontSize: 10, color: 'var(--ink-faint)' }}>send</span>
        <span className="kbd">⇧Enter</span>
        <span style={{ fontSize: 10, color: 'var(--ink-faint)' }}>newline</span>
      </div>
      <div className="cbox">
        {/* Stays editable while streaming — drafting the next message during a
            response is fine; send() guards against submitting mid-stream. */}
        <textarea
          ref={ref}
          value={text}
          onChange={grow}
          onKeyDown={onKeyDown}
          placeholder={streaming ? 'streaming… (draft your next message)' : 'message the engine, or / for a dev command'}
        />
        {mic.supported && (
          <button
            className={'send mic' + (mic.recording ? ' rec' : '')}
            title='hold to talk — e.g. "set a timer for 10 minutes"'
            onMouseDown={micStart}
            onMouseUp={micStop}
            onMouseLeave={() => mic.recording && micStop()}
            onTouchStart={(e) => {
              e.preventDefault()
              micStart()
            }}
            onTouchEnd={(e) => {
              e.preventDefault()
              micStop()
            }}
          >
            {mic.recording ? '● rec' : '🎙'}
          </button>
        )}
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
