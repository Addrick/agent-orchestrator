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
  const { perf, prefill, stale, sampledAt } = useKoboldPerf()
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

  // Live progress, where the sidecar is deployed. Only trust it while it agrees
  // that something is running: a stale `prefill` blob outliving its run would
  // otherwise pin a bar at 34% forever (the sidecar ages its own state out, but
  // this render must not depend on that being the only guard).
  //
  // `stale` is the second half of that guard. A failed poll keeps the last good
  // blob on purpose — a dropped request is not evidence the counters reset —
  // but "last known" is not "live", and rendering it anyway put the row in a
  // state that contradicted itself: `backend unreachable` in amber next to a
  // pulsing ingest bar frozen mid-run.
  const live = stale ? null : prefill
  const ingesting = !!live && live.phase === 'prefill' && (live.total ?? 0) > 0
  const ingestDone = live?.processed ?? 0
  const ingestTotal = live?.total ?? 0
  const ingestPct = ingestTotal ? Math.min(100, Math.round((ingestDone / ingestTotal) * 100)) : 0
  const decoding = !!live && live.phase === 'generate' && (live.generate_total ?? 0) > 0

  return (
    <div className="statusline" role="status">
      <span className={`sl-chip ${busy ? 'sl-busy' : ''} ${stale ? 'sl-stale' : ''}`}>
        <span className="dot" />
        {stale
          ? 'backend unreachable'
          : busy
            ? `${ingesting ? 'ingesting' : 'generating'} · ${fmtDur(since)}`
            : 'idle'}
        {perf.queue > 0 && ` · queue ${perf.queue}`}
      </span>

      {ingesting ? (
        /* Live, from the sidecar tailing KCPP's stdout — the only place these
           per-batch counts exist. Steps by blasbatchsize, so it advances in
           visible jumps rather than smoothly; that is the real granularity, not
           a rendering artifact. */
        <span
          className="sl-chip sl-busy"
          title="live prompt ingestion (kcpp-progress sidecar); steps by blasbatchsize"
        >
          ingest{' '}
          <b>
            {fmtTok(ingestDone)}/{fmtTok(ingestTotal)} tok
          </b>
          <span className="sl-bar" aria-hidden="true">
            <i style={{ width: `${ingestPct}%` }} />
          </span>
          <span className="sl-sub">{ingestPct}%</span>
        </span>
      ) : (
        <span
          className="sl-chip"
          title={
            busy
              ? 'Counters freeze during a generation — perf reports the previous run until this one completes.'
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
      )}

      {decoding ? (
        <span className="sl-chip sl-busy" title="live decode progress (kcpp-progress sidecar)">
          gen{' '}
          <b>
            {fmtTok(live?.generated ?? 0)}/{fmtTok(live?.generate_total ?? 0)} tok
          </b>
        </span>
      ) : (
        <span className="sl-chip" title="Last generation: decode tokens, and prefill + decode wall time.">
          gen {hasGen ? <b>{fmtTok(perf.last_token_count)} tok</b> : <b>—</b>}
          {hasGen && (
            <span className="sl-sub">
              {' '}
              · {perf.last_eval_speed.toFixed(1)} t/s · total <b>{fmtDur(genTotal)}</b>
            </span>
          )}
        </span>
      )}

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
