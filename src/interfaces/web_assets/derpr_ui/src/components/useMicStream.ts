import { useCallback, useEffect, useRef, useState } from 'react'
import { floatTo16, openMicGraph, type MicGraph } from './micGraph'

// Always-listening streaming dictation (DP-238 item B). A toggle opens a
// WebSocket to /voice/stream, then continuously streams mic frames (shared
// capture plumbing in micGraph.ts → int16) as binary messages. The server
// VAD-endpoints them and pushes back {text} per closed utterance, which
// `onText` routes to the composer (draft or auto-send, same as the
// hold-to-talk button). This is a continuous hot mic — only open while the
// toggle is on.
//
// Endpointing quality (mid-sentence cuts vs latency) is bounded by the server
// VAD; smarter turn-end detection is a follow-up (see tasks/DP-238.md).

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
  useEffect(() => {
    onTextRef.current = onText
  })

  const graphRef = useRef<MicGraph | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  // `active` only flips true at the end of the async start(), so two fast toggles
  // would both see active=false and open a second stream/ctx/ws that orphans the
  // first. This ref guards re-entry synchronously.
  const startingRef = useRef(false)

  const teardown = useCallback(() => {
    startingRef.current = false
    graphRef.current?.close()
    graphRef.current = null
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN)
      wsRef.current.close()
    wsRef.current = null
    setActive(false)
  }, [])

  const start = useCallback(async () => {
    if (startingRef.current || wsRef.current) return // already starting/active
    startingRef.current = true
    setError(null)

    let graph: MicGraph
    try {
      // Frames route through wsRef so the graph can open before the socket;
      // the readyState guard drops frames until the server is listening.
      graph = await openMicGraph((f) => {
        const w = wsRef.current
        if (!w || w.readyState !== WebSocket.OPEN) return
        w.send(floatTo16([f]).buffer as ArrayBuffer)
      })
    } catch {
      setError('mic access denied')
      startingRef.current = false
      return
    }

    const ws = new WebSocket(wsUrl('/voice/stream'))
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => ws.send(JSON.stringify({ sample_rate: graph.ctx.sampleRate }))
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

    graphRef.current = graph
    wsRef.current = ws
    startingRef.current = false
    setActive(true)
  }, [teardown])

  const toggle = useCallback(() => {
    if (active) teardown()
    else void start()
  }, [active, start, teardown])

  return { supported, active, error, toggle }
}
