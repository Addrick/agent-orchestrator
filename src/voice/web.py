# src/voice/web.py
"""Browser/phone push-to-talk capture for the voice pipeline (DP-238 web).

Discord voice *receive* is impossible under Discord's mandatory DAVE end-to-end
encryption (see memory codebase/dp238-discord-voice-recv-dead-dave-e2ee.md), so
the capture source for the "set a timer" command is a hold-to-talk button served
from the existing FastAPI web interface (:5003). The browser records mic audio
with the native Web Audio API (no extra packages), and on button release POSTs
one raw 16-bit PCM utterance to ``/voice/utterance``. The server resamples it to
16 kHz mono and routes it straight through STT → keyword intent → timer (no VAD:
the button already delimits the utterance).

``register_voice_web`` is the only public surface; the integration passes a
handler that drives ``VoicePipeline.submit_utterance`` and formats the reply.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

# handler(pcm_bytes, sample_rate) -> {"text": str, "message": str, "matched": bool}
UtteranceHandler = Callable[[bytes, int], Awaitable[Dict[str, Any]]]
# transcribe(pcm_bytes, sample_rate) -> {"text": str} (dictation, no intent routing)
TranscribeHandler = Callable[[bytes, int], Awaitable[Dict[str, Any]]]

# Cap an uploaded utterance so a stuck/abusive client can't post unbounded audio.
# 16-bit mono @ 48 kHz ≈ 96 KB/s, so 8 MB ≈ ~85 s of speech — plenty for a command.
_MAX_UTTERANCE_BYTES = 8 * 1024 * 1024


async def _dispatch(request: Request, fn: UtteranceHandler, what: str) -> JSONResponse:
    """Validate a raw-PCM upload (octet-stream body + ``X-Sample-Rate`` header),
    run ``fn``, and shape the JSON reply. Shared by both upload routes."""
    try:
        sample_rate = int(request.headers.get("X-Sample-Rate", "0"))
    except ValueError:
        sample_rate = 0
    if sample_rate <= 0:
        return JSONResponse({"error": "missing/invalid X-Sample-Rate"}, status_code=400)
    pcm = await request.body()
    if not pcm:
        return JSONResponse({"error": "empty body"}, status_code=400)
    if len(pcm) > _MAX_UTTERANCE_BYTES:
        return JSONResponse({"error": "utterance too large"}, status_code=413)
    try:
        result = await fn(pcm, sample_rate)
    except Exception:  # noqa: BLE001 - a bad upload must not 500 the whole app
        logger.exception("voice web %s handler failed", what)
        return JSONResponse({"error": "internal error"}, status_code=500)
    return JSONResponse(result)


def register_voice_web(
    app: FastAPI,
    handler: UtteranceHandler,
    transcribe_handler: TranscribeHandler,
) -> None:
    """Mount the push-to-talk page + upload routes on an existing FastAPI app.

    Two upload routes share the same raw-PCM body contract: ``/voice/utterance``
    runs the full STT→intent→timer path (the standalone ``/voice`` page), while
    ``/voice/transcribe`` is STT-only for the SPA mic button (dictation into the
    composer — the LLM owns intents)."""

    @app.get("/voice", response_class=HTMLResponse)
    async def _voice_page() -> HTMLResponse:  # pragma: no cover - static asset
        return HTMLResponse(_PTT_PAGE)

    @app.post("/voice/utterance")
    async def _voice_utterance(request: Request) -> JSONResponse:
        return await _dispatch(request, handler, "utterance")

    @app.post("/voice/transcribe")
    async def _voice_transcribe(request: Request) -> JSONResponse:
        return await _dispatch(request, transcribe_handler, "transcribe")

    logger.info("Voice push-to-talk web capture mounted at GET /voice")


# --- the push-to-talk page (inlined to avoid static-asset packaging paths) ----
# Captures mic audio via getUserMedia + a ScriptProcessorNode (universally
# supported, no dependency), converts Float32 → little-endian int16 mono, and
# POSTs the raw buffer with the AudioContext's native sample rate in a header.
# Server-side ``to_16k_mono`` does the resample, so the browser stays dumb.
_PTT_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>derpr voice</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; background: #15171c; color: #e6e6e6;
         display: flex; flex-direction: column; align-items: center; gap: 1.25rem;
         padding: 2rem 1rem; margin: 0; }
  h1 { font-size: 1.1rem; font-weight: 600; margin: 0; color: #9aa4b2; }
  #talk { width: 220px; height: 220px; border-radius: 50%; border: none;
          font-size: 1.1rem; font-weight: 600; color: #fff; background: #3b82f6;
          cursor: pointer; user-select: none; -webkit-user-select: none;
          touch-action: none; transition: background .1s, transform .1s; }
  #talk:active, #talk.rec { background: #ef4444; transform: scale(.97); }
  #talk:disabled { background: #444; cursor: default; }
  #status { min-height: 1.2rem; color: #9aa4b2; font-size: .9rem; }
  #out { max-width: 30rem; width: 100%; }
  .line { background: #1e2128; border-radius: .5rem; padding: .6rem .8rem;
          margin: .4rem 0; }
  .t { color: #cbd5e1; } .m { color: #86efac; } .e { color: #fca5a5; }
</style>
</head>
<body>
  <h1>hold to talk — e.g. "set a timer for 10 minutes"</h1>
  <button id="talk">hold</button>
  <div id="status">tap the button to grant mic access</div>
  <div id="out"></div>
<script>
const talk = document.getElementById('talk');
const statusEl = document.getElementById('status');
const out = document.getElementById('out');
let ctx, stream, source, node, chunks = [], recording = false;

function log(text, cls) {
  const d = document.createElement('div');
  d.className = 'line ' + (cls || '');
  d.textContent = text;
  out.prepend(d);
}

async function ensureAudio() {
  if (ctx) return;
  stream = await navigator.mediaDevices.getUserMedia({ audio: {
    channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  source = ctx.createMediaStreamSource(stream);
  node = ctx.createScriptProcessor(4096, 1, 1);
  node.onaudioprocess = (e) => {
    if (!recording) return;
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(node);
  node.connect(ctx.destination);
}

function floatTo16(chunks) {
  let len = 0;
  for (const c of chunks) len += c.length;
  const buf = new Int16Array(len);
  let off = 0;
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++) {
      let s = Math.max(-1, Math.min(1, c[i]));
      buf[off++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
  }
  return buf;
}

async function start() {
  try {
    await ensureAudio();
    if (ctx.state === 'suspended') await ctx.resume();
  } catch (err) {
    statusEl.textContent = 'mic access denied';
    return;
  }
  chunks = [];
  recording = true;
  talk.classList.add('rec');
  talk.textContent = 'listening…';
  statusEl.textContent = 'release to send';
}

async function stop() {
  if (!recording) return;
  recording = false;
  talk.classList.remove('rec');
  talk.textContent = 'hold';
  const pcm = floatTo16(chunks);
  chunks = [];
  if (pcm.length < ctx.sampleRate * 0.2) {  // < 0.2s — likely a misclick
    statusEl.textContent = 'too short';
    return;
  }
  statusEl.textContent = 'sending…';
  talk.disabled = true;
  try {
    const r = await fetch('/voice/utterance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream',
                 'X-Sample-Rate': String(ctx.sampleRate) },
      body: pcm.buffer,
    });
    const j = await r.json();
    if (!r.ok || j.error) { log(j.error || ('error ' + r.status), 'e'); }
    else {
      if (j.text) log('you: ' + j.text, 't');
      if (j.message) log(j.message, 'm');
    }
    statusEl.textContent = 'ready';
  } catch (err) {
    log('network error', 'e');
    statusEl.textContent = 'ready';
  } finally {
    talk.disabled = false;
  }
}

talk.addEventListener('mousedown', start);
talk.addEventListener('mouseup', stop);
talk.addEventListener('mouseleave', () => { if (recording) stop(); });
talk.addEventListener('touchstart', (e) => { e.preventDefault(); start(); }, { passive: false });
talk.addEventListener('touchend', (e) => { e.preventDefault(); stop(); }, { passive: false });
</script>
</body>
</html>
"""
