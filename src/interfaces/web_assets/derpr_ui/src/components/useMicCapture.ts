import { useCallback, useEffect, useRef, useState } from 'react'
import { floatTo16, openMicGraph, type MicGraph } from './micGraph'

// Hold-to-talk mic capture for the composer (DP-238). Ports the proven capture
// path from the standalone /voice page (src/voice/web.py): getUserMedia →
// ScriptProcessorNode → Float32 → little-endian int16 mono (shared plumbing in
// micGraph.ts). Capture only — the caller POSTs the buffer to /voice/transcribe
// and decides what to do with the text (draft vs auto-send).

export interface MicChunk {
  pcm: ArrayBuffer
  sampleRate: number
}

export interface MicCapture {
  supported: boolean
  recording: boolean
  error: string | null
  // Resolves true once recording, false if mic access failed (so the caller can
  // surface the denial — `error` is set on the same render, too stale to read).
  start: () => Promise<boolean>
  stop: () => Promise<MicChunk | null>
}

const MIN_SECONDS = 0.2 // shorter than this is almost certainly a misclick

export function useMicCapture(): MicCapture {
  const supported =
    typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const graphRef = useRef<MicGraph | null>(null)
  const chunksRef = useRef<Float32Array[]>([])
  const recordingRef = useRef(false)

  // Stop the mic and free the graph. Released after every utterance (not kept
  // warm) so the OS/browser "recording" indicator turns off between clips and
  // the ScriptProcessor isn't left firing for the whole session; the next press
  // rebuilds it (getUserMedia won't re-prompt once granted).
  const teardown = useCallback(() => {
    graphRef.current?.close()
    graphRef.current = null
  }, [])

  // Release the mic if the component unmounts mid-record.
  useEffect(() => teardown, [teardown])

  const start = useCallback(async (): Promise<boolean> => {
    setError(null)
    try {
      if (!graphRef.current) {
        graphRef.current = await openMicGraph((f) => {
          if (recordingRef.current) chunksRef.current.push(new Float32Array(f))
        })
      }
    } catch {
      setError('mic access denied')
      teardown()
      return false
    }
    chunksRef.current = []
    recordingRef.current = true
    setRecording(true)
    return true
  }, [teardown])

  const stop = useCallback(async (): Promise<MicChunk | null> => {
    if (!recordingRef.current) return null
    recordingRef.current = false
    setRecording(false)
    const sampleRate = graphRef.current?.ctx.sampleRate ?? 0
    const chunks = chunksRef.current
    chunksRef.current = []
    teardown()
    if (!sampleRate) return null
    const pcm = floatTo16(chunks)
    if (pcm.length < sampleRate * MIN_SECONDS) return null // misclick
    return { pcm: pcm.buffer as ArrayBuffer, sampleRate }
  }, [teardown])

  return { supported, recording, error, start, stop }
}
