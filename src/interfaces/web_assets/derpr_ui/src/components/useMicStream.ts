import { useCallback, useRef, useState } from 'react'

// Always-listening streaming dictation (DP-238 item B). A toggle opens a
// WebSocket to /voice/stream, then continuously streams mic frames (getUserMedia
// → ScriptProcessorNode → int16) as binary messages. The server VAD-endpoints
// them and pushes back {text} per closed utterance, which `onText` routes to the
// composer (draft or auto-send, same as the hold-to-talk button). This is a
// continuous hot mic — only open while the toggle is on.
//
// ScriptProcessorNode is deprecated; an AudioWorklet (off the audio thread) is
// the upgrade for drop-free capture under load — a known follow-up. Endpointing
// quality (mid-sentence cuts vs latency) is bounded by the server VAD; smarter
// turn-end detection is also a follow-up (see tasks/DP-238.md).

export interface MicStream {
  supported: boolean
  active: boolean
  error: string | null
  toggle: () => void
}

function wsUrl(path: string): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${location.host}${path}`
}

export function useMicStream(onText: (t: string) => void): MicStream {
  const supported =
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof WebSocket !== 'undefined'
  const [active, setActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // onText changes identity each render; read the latest via a ref so the audio
  // callback closure never goes stale.
  const onTextRef = useRef(onText)
  onTextRef.current = onText

  const ctxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const nodeRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const teardown = useCallback(() => {
    nodeRef.current?.disconnect()
    sourceRef.current?.disconnect()
    streamRef.current?.getTracks().forEach((t) => t.stop())
    if (ctxRef.current && ctxRef.current.state !== 'closed') ctxRef.current.close()
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN)
      wsRef.current.close()
    nodeRef.current = null
    sourceRef.current = null
    streamRef.current = null
    ctxRef.current = null
    wsRef.current = null
    setActive(false)
  }, [])

  const start = useCallback(async () => {
    setError(null)
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      })
    } catch {
      setError('mic access denied')
      return
    }
    const Ctx: typeof AudioContext =
      window.AudioContext || (window as any).webkitAudioContext
    const ctx = new Ctx()
    if (ctx.state === 'suspended') await ctx.resume()
    const source = ctx.createMediaStreamSource(stream)
    const node = ctx.createScriptProcessor(4096, 1, 1)

    const ws = new WebSocket(wsUrl('/voice/stream'))
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => ws.send(JSON.stringify({ sample_rate: ctx.sampleRate }))
    ws.onmessage = (e) => {
      try {
        const t = (JSON.parse(e.data).text as string)?.trim()
        if (t) onTextRef.current(t)
      } catch {
        /* ignore malformed frame */
      }
    }
    ws.onerror = () => setError('stream connection error')
    ws.onclose = () => {
      if (wsRef.current === ws) teardown()
    }

    node.onaudioprocess = (ev) => {
      if (ws.readyState !== WebSocket.OPEN) return
      const f = ev.inputBuffer.getChannelData(0)
      const buf = new Int16Array(f.length)
      for (let i = 0; i < f.length; i++) {
        const s = Math.max(-1, Math.min(1, f[i]))
        buf[i] = s < 0 ? s * 0x8000 : s * 0x7fff
      }
      ws.send(buf.buffer)
    }
    source.connect(node)
    node.connect(ctx.destination)

    streamRef.current = stream
    ctxRef.current = ctx
    sourceRef.current = source
    nodeRef.current = node
    wsRef.current = ws
    setActive(true)
  }, [teardown])

  const toggle = useCallback(() => {
    if (active) teardown()
    else void start()
  }, [active, start, teardown])

  return { supported, active, error, toggle }
}
