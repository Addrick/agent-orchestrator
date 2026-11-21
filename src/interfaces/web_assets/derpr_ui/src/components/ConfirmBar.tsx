import type { ConfirmStatus } from '../state/store'

interface Props {
  token: string | null | undefined
  status: ConfirmStatus | undefined
  onResolve: (token: string, approved: boolean) => void
}

/** Approve/deny bar for one proposed write (6a, DP-528). Shared by the
 *  persisted ephemeral chunk and the live streaming row, so a write looks and
 *  answers the same before and after its turn ends. A click while a reply
 *  streams is queued, not dropped; clicking the queued answer again undoes it. */
export function ConfirmBar({ token, status, onResolve }: Props) {
  const resolving = status === 'resolving'
  const disabled = resolving || !token
  const note = resolving
    ? 'resolving…'
    : status
      ? 'queued · sent when the reply finishes · click again to undo'
      : 'resolves via /confirm · tool_policy = CONFIRM'
  return (
    <div className="confirm">
      <span className="lbl">CONFIRM</span>
      <button
        className={'btn approve' + (status === 'approve' ? ' queued' : '')}
        disabled={disabled}
        title="approve & run the parked write"
        onClick={() => token && onResolve(token, true)}
      >
        {status === 'approve' ? '✓ approve queued' : '✓ approve & run'}
      </button>
      <button
        className={'btn deny' + (status === 'deny' ? ' queued' : '')}
        disabled={disabled}
        title="deny the parked write"
        onClick={() => token && onResolve(token, false)}
      >
        {status === 'deny' ? '✕ deny queued' : '✕ deny'}
      </button>
      <span className="note">{note}</span>
    </div>
  )
}
