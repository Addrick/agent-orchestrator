import { useCallback, useRef, useState } from 'react'

// Hold-to-talk mic capture for the composer (DP-238). Ports the proven capture
// path from the standalone /voice page (src/voice/web.py): getUserMedia →
// ScriptProcessorNode → Float32 → little-endian int16 mono. Capture only — the
// caller POSTs the buffer to /voice/transcribe and decides what to do with the
// text (draft vs auto-send). ScriptProcessorNode is deprecated but universally
// supported and adequate for short push-to-talk clips (no AudioWorklet needed
// until the streaming/always-listening path — DP-238 item B).

export interface MicChunk {
  pcm: ArrayBuffer
  sampleRate: number
}

export interface MicCapture {
  supported: boolean
  recording: boolean
  error: string | null
  start: () => Promise<void>
  stop: () => Promise<MicChunk | null>
}

const MIN_SECONDS = 0.2 // shorter than this is almost certainly a misclick

function floatTo16(chunks: Float32Array[]): Int16Array {
  let len = 0
  for (const c of chunks) len += c.length
  const buf = new Int16Array(len)
  let off = 0
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++) {
      const s = Math.max(-1, Math.min(1, c[i]))
      buf[off++] = s < 0 ? s * 0x8000 : s * 0x7fff
    }
  }
  return buf
}

export function useMicCapture(): MicCapture {
  const supported =
    typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const ctxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const nodeRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const chunksRef = useRef<Float32Array[]>([])
  const recordingRef = useRef(false)

  const ensureAudio = useCallback(async () => {
    if (ctxRef.current) return
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    })
    const Ctx: typeof AudioContext =
      window.AudioContext || (window as any).webkitAudioContext
    const ctx = new Ctx()
    const source = ctx.createMediaStreamSource(stream)
    const node = ctx.createScriptProcessor(4096, 1, 1)
    node.onaudioprocess = (e) => {
      if (!recordingRef.current) return
      chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    source.connect(node)
    node.connect(ctx.destination)
    streamRef.current = stream
    ctxRef.current = ctx
    sourceRef.current = source
    nodeRef.current = node
  }, [])

  const start = useCallback(async () => {
    setError(null)
    try {
      await ensureAudio()
      if (ctxRef.current!.state === 'suspended') await ctxRef.current!.resume()
    } catch {
      setError('mic access denied')
      return
    }
    chunksRef.current = []
    recordingRef.current = true
    setRecording(true)
  }, [ensureAudio])

  const stop = useCallback(async (): Promise<MicChunk | null> => {
    if (!recordingRef.current) return null
    recordingRef.current = false
    setRecording(false)
    const ctx = ctxRef.current
    if (!ctx) return null
    const chunks = chunksRef.current
    chunksRef.current = []
    const pcm = floatTo16(chunks)
    if (pcm.length < ctx.sampleRate * MIN_SECONDS) return null // misclick
    return { pcm: pcm.buffer as ArrayBuffer, sampleRate: ctx.sampleRate }
  }, [])

  return { supported, recording, error, start, stop }
}
