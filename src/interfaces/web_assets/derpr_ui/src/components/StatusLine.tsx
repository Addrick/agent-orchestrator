/* DP-311 — backend processing statusline.
 *
 * One row under the topbar showing what the KoboldCPP backend is doing:
 * prompt-token ingestion, total time of the last generation, and why/when it
 * stopped. Sourced from `GET /api/extra/perf` (see useKoboldPerf for why that
 * is the only remote source, and what it cannot tell us).
 */
import { useEffect, useState } from 'react'
import {
  useKoboldPerf,
  stopReasonLabel,
  fmtDur,
  fmtAgo,
  clockAt,
} from '../state/useKoboldPerf'
import { fmtTok } from '../state/util'

/** Wall clock that re-renders on an interval, so the busy elapsed timer and the
 *  "N ago" stamp advance between polls. Kept in state rather than read during
 *  render — `Date.now()` in a render body is impure and makes the output depend
 *  on when React happens to re-render. Returns 0 before the first tick. */
function useNow(active: boolean, ms: number): number {
  const [now, setNow] = useState(0)
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(t)
  }, [active, ms])
  return now
}

export function StatusLine() {
  const { perf, stale, sampledAt } = useKoboldPerf()
  const busy = !!perf && perf.idle === 0
  const tick = useNow(!!perf, busy ? 250 : 5000)
  // Before the first tick lands, anchor on the sample itself so nothing renders
  // a 1970 timestamp for a frame.
  const now = tick || sampledAt

  if (!perf) {
    return (
      <div className="statusline" role="status">
        <span className="sl-chip sl-muted">
          <span className="dot" /> backend · {stale ? 'unreachable' : 'connecting…'}
        </span>
      </div>
    )
  }

  // `idletime` is sampled server-side; extrapolate with the wall clock so the
  // reading does not visibly freeze between polls.
  const since = perf.idletime + Math.max(0, (now - sampledAt) / 1000)
  const genTotal = perf.last_process_time + perf.last_eval_time
  // `total_gens` counts COMPLETED generations, so it is the honest "is there a
  // last gen to report" test. `stop_reason !== -1` is not: it flips to 0 the
  // moment the first generation starts, before any counter is filled in.
  const hasGen = perf.total_gens > 0

  return (
    <div className="statusline" role="status">
      <span className={`sl-chip ${busy ? 'sl-busy' : ''} ${stale ? 'sl-stale' : ''}`}>
        <span className="dot" />
        {stale ? 'backend unreachable' : busy ? `generating · ${fmtDur(since)}` : 'idle'}
        {perf.queue > 0 && ` · queue ${perf.queue}`}
      </span>

      <span
        className="sl-chip"
        title={
          busy
            ? 'Counters freeze during a generation — KCPP exposes live prefill progress only in its stdout log, not over the API.'
            : 'Prompt tokens ingested by the last completed generation (KCPP last_input_count).'
        }
      >
        ingest{' '}
        {hasGen ? (
          <b>
            {fmtTok(perf.last_input_count)}/{fmtTok(perf.last_input_count)} tok
          </b>
        ) : (
          <b>—</b>
        )}
        {hasGen && perf.last_process_time > 0 && (
          <span className="sl-sub">
            {' '}
            · {fmtDur(perf.last_process_time)} · {perf.last_process_speed.toFixed(0)} t/s
          </span>
        )}
      </span>

      <span className="sl-chip" title="Last generation: decode tokens, and prefill + decode wall time.">
        gen {hasGen ? <b>{fmtTok(perf.last_token_count)} tok</b> : <b>—</b>}
        {hasGen && (
          <span className="sl-sub">
            {' '}
            · {perf.last_eval_speed.toFixed(1)} t/s · total <b>{fmtDur(genTotal)}</b>
          </span>
        )}
      </span>

      {/* Two reasons this chip goes blank mid-generation, both verified against a
          live KCPP rather than assumed:
          1. `stop_reason` is RESET TO 0 when a generation starts, so a running
             gen reads as "out of tokens" — showing it would report a stop that
             has not happened (and overwrite the previous run's real reason).
          2. `idletime` measures elapsed-in-generation while busy, not the age of
             the last stop, so neither "ago" nor a wall-clock stamp is derivable. */}
      <span
        className="sl-chip"
        title={
          busy
            ? 'stop_reason resets when a generation starts — the real value lands when this run completes'
            : hasGen
              ? `last generation ended at ${clockAt(now, since)}`
              : 'no generation yet this backend uptime'
        }
      >
        stop:{' '}
        {busy ? (
          <b>…</b>
        ) : hasGen ? (
          <>
            <b>{stopReasonLabel(perf.stop_reason)}</b>
            <span className="sl-sub"> · {fmtAgo(since)}</span>
          </>
        ) : (
          <b>—</b>
        )}
      </span>

      <span className="sl-spacer" />
      <span className="sl-chip sl-muted" title="backend uptime and completed generations">
        {perf.total_gens} gens · up {fmtAgo(perf.uptime).replace(' ago', '')}
      </span>
    </div>
  )
}
