// Shared mic-capture plumbing for useMicCapture (hold-to-talk) and useMicStream
// (always-listening). One place for the getUserMedia constraints, the
// webkitAudioContext shim, the ScriptProcessorNode wiring, and teardown — the
// planned ScriptProcessor→AudioWorklet migration happens here once, not per hook.
// Callers keep only their routing (buffer-and-POST vs WS-send).

export interface MicGraph {
  ctx: AudioContext
  close: () => void
}

// Opens mic + audio graph and calls `onFrame` with each raw capture frame.
// NOTE: the Float32Array is the live channel buffer — copy it if you keep it
// past the callback (`new Float32Array(f)`).
export async function openMicGraph(
  onFrame: (frame: Float32Array) => void,
): Promise<MicGraph> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  })
  try {
    const Ctx: typeof AudioContext =
      window.AudioContext ||
      (window as Window & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext!
    const ctx = new Ctx()
    if (ctx.state === 'suspended') await ctx.resume()
    const source = ctx.createMediaStreamSource(stream)
    const node = ctx.createScriptProcessor(4096, 1, 1)
    node.onaudioprocess = (e) => onFrame(e.inputBuffer.getChannelData(0))
    source.connect(node)
    node.connect(ctx.destination)
    let closed = false
    const close = () => {
      if (closed) return
      closed = true
      node.disconnect()
      source.disconnect()
      stream.getTracks().forEach((t) => t.stop())
      if (ctx.state !== 'closed') void ctx.close()
    }
    return { ctx, close }
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop())
    throw e
  }
}

// Float32 [-1,1] → little-endian int16 mono, concatenating all chunks.
export function floatTo16(chunks: Float32Array[]): Int16Array {
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
