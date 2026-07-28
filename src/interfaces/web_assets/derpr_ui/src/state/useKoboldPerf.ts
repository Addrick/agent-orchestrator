/* ============================================================
   DP-311 — KoboldCPP processing state for the statusline.

   Polls `GET /api/extra/perf` (proxied by the adapter to the backend KCPP).
   That endpoint is the only remote source of KCPP's processing state; the
   per-step "Processing Prompt (x / y tokens)" progress exists ONLY in KCPP's
   stdout log, which is block-buffered, host-local, and lags by minutes — so
   the statusline reports the LAST COMPLETED gen's ingestion counts plus a live
   busy/elapsed indicator, not a live prefill bar.

   The poll is backend-wide, not per-tab: a generation started by Discord, an
   agent, or another browser shows up here too. That is the point — this line
   answers "what is the box doing", not "what is my tab doing".
   ============================================================ */
import { useEffect, useRef, useState } from 'react'
import { getKoboldPerf, getKoboldPrefill } from '../api/client'
import type { KoboldPerf, KoboldPrefill } from '../types/contracts'

/** Poll cadence. Busy is polled hard enough for the elapsed clock to read as
 *  live; idle is slow because nothing changes until a gen starts, and this
 *  endpoint is hit by every open portal tab. */
const BUSY_MS = 1000
const IDLE_MS = 5000

export interface PerfSnapshot {
  /** Last successful sample, or null before the first one lands. */
  perf: KoboldPerf | null
  /** Live ingestion progress from the kcpp-progress sidecar, or null where it
   *  is not deployed (the common case — poll stops after the first such
   *  answer). Present alongside `perf`, never instead of it. */
  prefill: KoboldPrefill | null
  /** True once a poll has failed since the last success — the numbers shown
   *  are the last good ones and may be arbitrarily old. */
  stale: boolean
  /** Wall-clock ms at which `perf` was received. Needed because `idletime` is
   *  measured at sample time and keeps running afterwards. */
  sampledAt: number
}

const EMPTY: PerfSnapshot = { perf: null, prefill: null, stale: false, sampledAt: 0 }

export function useKoboldPerf(): PerfSnapshot {
  const [snap, setSnap] = useState<PerfSnapshot>(EMPTY)
  // The cadence must react to the busy flag we just read, but rescheduling off
  // a state dep would tear down and rebuild the timer on every sample. Keep the
  // loop self-scheduling and read the flag from a ref.
  const busyRef = useRef(false)
  // Set once the engine answers `not_configured` — that verdict cannot change
  // without an engine restart, so polling a route that will keep saying "no
  // sidecar here" every second is pure waste.
  const noSidecarRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const ctl = new AbortController()

    const tick = async () => {
      try {
        const perf = await getKoboldPerf(ctl.signal)
        if (cancelled) return
        busyRef.current = perf.idle === 0

        // Progress is best-effort and secondary: its failure must never cost us
        // the perf sample we already hold, so it is fetched and caught
        // separately rather than in the same try.
        let prefill: KoboldPrefill | null = null
        if (!noSidecarRef.current) {
          try {
            const p = await getKoboldPrefill(ctl.signal)
            if (p.available) {
              prefill = p
            } else if (p.reason === 'not_configured') {
              noSidecarRef.current = true
            }
          } catch {
            /* transient — fall back to perf's last-completed counters */
          }
        }
        if (cancelled) return
        setSnap({ perf, prefill, stale: false, sampledAt: Date.now() })
      } catch {
        if (cancelled) return
        // Keep the last good sample — a dropped poll is not evidence that the
        // backend's counters reset, only that we could not read them.
        busyRef.current = false
        setSnap((s) => (s.stale ? s : { ...s, stale: true }))
      }
      if (cancelled) return
      timer = setTimeout(tick, busyRef.current ? BUSY_MS : IDLE_MS)
    }

    void tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
      ctl.abort()
    }
  }, [])

  return snap
}

/* ---- formatting (pure, exported for reuse/testing) ------------------- */

/** KCPP stop_reason codes. 0/1/2 verified against a live KCPP 1.115; 3 is the
 *  abort path (what `/api/extra/abort` produces); -1 means "no gen yet". */
export function stopReasonLabel(code: number): string {
  switch (code) {
    case 0:
      return 'out of tokens'
    case 1:
      return 'EOS'
    case 2:
      return 'stop sequence'
    case 3:
      return 'aborted'
    case -1:
      return 'none yet'
    default:
      return `code ${code}`
  }
}

/** Compact duration: sub-minute in seconds, above that m/s. */
export function fmtDur(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '—'
  if (seconds < 10) return `${seconds.toFixed(2)}s`
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  return `${m}m${String(Math.round(seconds % 60)).padStart(2, '0')}s`
}

/** "how long ago", for the last stop's timestamp. */
export function fmtAgo(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  return `${Math.round(seconds / 3600)}h ago`
}

/** Absolute clock time of an event that happened `agoSeconds` before `now`. */
export function clockAt(now: number, agoSeconds: number): string {
  return new Date(now - agoSeconds * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}
