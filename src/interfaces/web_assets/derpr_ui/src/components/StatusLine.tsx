/* DP-311 — backend processing statusline.
 *
 * One row under the topbar showing what the KoboldCPP backend is doing:
 * prompt-token ingestion, total time of the last generation, and why/when it
 * stopped. Sourced from `GET /api/extra/perf` (see useKoboldPerf for why that
 * is the only remote source, and what it cannot tell us).
 */
import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { InferenceJob, InferenceServer, KoboldPerf, KoboldPrefill } from '../types/contracts'
import {
  serversBusy,
  useKoboldPerf,
  stopReasonLabel,
  fmtDur,
  fmtAgo,
  clockAt,
} from '../state/useKoboldPerf'
import { fmtTok } from '../state/util'
import { LocalBackendSwitch } from './LocalBackendSwitch'

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
  const { perf, prefill, stale, sampledAt, servers, refresh } = useKoboldPerf()
  const backendSwitch = <LocalBackendSwitch onSwitched={refresh} />
  const busy = !!perf && perf.idle === 0
  const tick = useNow(!!perf || !!servers, busy || (!!servers && serversBusy(servers)) ? 250 : 5000)
  // Before the first tick lands, anchor on the sample itself so nothing renders
  // a 1970 timestamp for a frame.
  const now = tick || sampledAt

  if (servers) {
    // DP-499: one row per box that is up. The primary keeps its row while down
    // so an outage stays visible; any box with work on it keeps one too.
    const rows = (servers.servers ?? []).filter((b) => b.healthy || b.primary || b.jobs.length > 0)
    const held = servers.held ?? []
    const late = Math.max(0, (now - sampledAt) / 1000)
    return (
      <div className="statusline-stack">
        {rows.map((b, i) => (
          <ServerRow
            key={b.name}
            server={b}
            lead={i === 0 ? backendSwitch : null}
            tail={
              i === 0 && held.length > 0 ? (
                <span
                  className="sl-chip sl-busy"
                  title="chat requests the router is holding for a free box, oldest first"
                >
                  waiting: <b>{held.map((j) => `${j.requester} ${fmtDur(j.elapsed_s + late)}`).join(', ')}</b>
                </span>
              ) : null
            }
            sampledAt={sampledAt}
            stale={stale}
            now={now}
          />
        ))}
      </div>
    )
  }

  if (!perf) {
    return (
      <div className="statusline" role="status">
        {backendSwitch}
        <span className="sl-chip sl-muted">
          <span className="dot" /> backend · {stale ? 'unreachable' : 'connecting…'}
        </span>
      </div>
    )
  }

  return (
    <div className="statusline" role="status">
      {backendSwitch}
      <PerfChips perf={perf} prefill={prefill} stale={stale} sampledAt={sampledAt} now={now} />
    </div>
  )
}

/** A row's sample older than this (about three missed router polls) is shown as stale. */
const SAMPLE_STALE_S = 10

/** Label for who a job is for. A background job on the router is Hindsight's
 *  unless its client named itself. */
function jobLabel(j: InferenceJob): string {
  return j.kind === 'background' ? `${j.requester} job` : j.requester
}

/** DP-499: one box behind the router — its name, who its work is for, then the
 *  same chips as the single-server line, read from that box. */
function ServerRow({
  server,
  lead,
  tail,
  sampledAt,
  stale,
  now,
}: {
  server: InferenceServer
  lead: ReactNode
  tail: ReactNode
  sampledAt: number
  stale: boolean
  now: number
}) {
  const [job, ...more] = server.jobs
  const late = Math.max(0, (now - sampledAt) / 1000)
  // The counters are the router's sample, taken `sample_age_s` before it
  // answered: anchor their clocks there, so a row's elapsed agrees with its job
  // chip. Older than a few missed polls (3 s each) is not live any more.
  const age = server.sample_age_s ?? 0
  const perfAt = sampledAt - age * 1000
  const perfStale = stale || age > SAMPLE_STALE_S
  return (
    <div className="statusline" role="status">
      {lead}
      <span
        className={`sl-chip sl-server ${server.primary ? 'sl-primary' : ''}`}
        title={`${server.name}${server.primary ? ' (primary)' : ''}${server.model ? ` · ${server.model}` : ''}`}
      >
        <b>{server.name}</b>
      </span>
      {server.state === 'unloaded' ? (
        <span className="sl-chip sl-muted" title="the server answers, but its model is not loaded; the router will not load it">
          <span className="dot" /> model not loaded
        </span>
      ) : !server.healthy ? (
        <span className="sl-chip sl-stale">
          <span className="dot" /> down
        </span>
      ) : (
        <>
          {job ? (
            <span
              className="sl-chip sl-busy"
              title={`${job.kind === 'background' ? 'background job' : 'chat request'} for ${job.requester}, running ${fmtDur(job.elapsed_s + late)}`}
            >
              for <b>{jobLabel(job)}</b>
              {more.length > 0 && <span className="sl-sub"> +{more.length}</span>}
            </span>
          ) : server.external_busy ? (
            <span className="sl-chip sl-busy" title="busy with work that did not come through the router">
              for <b>external</b>
            </span>
          ) : null}
          {server.perf ? (
            <PerfChips perf={server.perf} prefill={server.prefill} stale={perfStale} sampledAt={perfAt} now={now} />
          ) : (
            <span className="sl-chip sl-muted">no metrics from this server</span>
          )}
        </>
      )}
      {tail}
    </div>
  )
}

/** The single-server chips: state, ingest, gen, stop, gens/uptime. */
function PerfChips({
  perf,
  prefill,
  stale,
  sampledAt,
  now,
}: {
  perf: KoboldPerf
  prefill: KoboldPrefill | null
  stale: boolean
  sampledAt: number
  now: number
}) {
  const busy = perf.idle === 0
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
  const decoding = !!live && live.phase === 'generate' && ((live.generate_total ?? 0) > 0 || (live.generated ?? 0) > 0)

  return (
    <>
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
          title="live prompt ingestion (kcpp-progress sidecar or Strata /metrics); steps by blasbatchsize"
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
              ? (live?.total ?? 0) > 0
                ? 'Prompt tokens ingested for the active generation.'
                : 'Counters freeze during a generation — perf reports the previous run until this one completes.'
              : 'Prompt tokens ingested by the last completed generation (KCPP last_input_count).'
          }
        >
          ingest{' '}
          {busy && (live?.total ?? 0) > 0 ? (
            <b>
              {fmtTok(live?.processed || live?.total || 0)}/{fmtTok(live?.total || 0)} tok
            </b>
          ) : hasGen ? (
            <b>
              {fmtTok(perf.last_input_count)}/{fmtTok(perf.last_input_count)} tok
            </b>
          ) : (
            <b>—</b>
          )}
          {hasGen && perf.last_process_time > 0 && !busy && (
            <span className="sl-sub">
              {' '}
              · {fmtDur(perf.last_process_time)} · {perf.last_process_speed.toFixed(0)} t/s
            </span>
          )}
        </span>
      )}

      {decoding ? (
        <span className="sl-chip sl-busy" title="live decode progress">
          gen{' '}
          <b>
            {(live?.generate_total ?? 0) > 0
              ? `${fmtTok(live?.generated ?? 0)}/${fmtTok(live?.generate_total ?? 0)}`
              : fmtTok(live?.generated ?? 0)}{' '}
            tok
          </b>
          {((live?.tok_s != null && live.tok_s > 0) || perf.last_eval_speed > 0) && (
            <span className="sl-sub">
              {' '}
              · {(live?.tok_s && live.tok_s > 0 ? live.tok_s : perf.last_eval_speed).toFixed(1)} t/s
            </span>
          )}
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
    </>
  )
}
