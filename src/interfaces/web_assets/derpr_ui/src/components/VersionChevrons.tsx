import { useState, useEffect } from 'react'
import * as api from '../api/client'
import type { VersionsResponse } from '../types/contracts'

interface Props {
  interactionId: number
  onResync: () => void // re-fetch transcript after a version swap
}

/** Version chevrons ‹k/n›. Shown ONLY when the chunk has_versions.
 *  Lazily fetches GET /interaction/{id}/versions; canonical is the LAST
 *  entry per contract. On chevron click POSTs select_version/{k} (0-indexed
 *  archive position pre-swap), then re-syncs chevrons from the response and
 *  the transcript from the parent. */
export function VersionChevrons({ interactionId, onResync }: Props) {
  const [vers, setVers] = useState<VersionsResponse | null>(null)
  // current displayed version, 1-based; defaults to canonical (last).
  const [cur, setCur] = useState<number>(0)

  useEffect(() => {
    let live = true
    api.getVersions(interactionId)
      .then((v) => {
        if (!live) return
        setVers(v)
        // Find the 1-indexed position of the canonical version
        const canonicalIdx = v.versions.findIndex((entry) => entry.canonical)
        setCur(canonicalIdx !== -1 ? canonicalIdx + 1 : v.versions.length)
      })
      // Engine unreachable mid-session (client rethrows past mock fallback):
      // leave the disabled "…" chevrons rather than crash on an unhandled
      // rejection.
      .catch(() => {})
    return () => {
      live = false
    }
  }, [interactionId])

  if (!vers || vers.versions.length === 0) {
    return (
      <span className="chev">
        <button disabled>‹</button>
        <span className="ct">…</span>
        <button disabled>›</button>
      </span>
    )
  }

  const total = vers.versions.length

  const select = async (target1: number) => {
    const k = target1 - 1 // 0-indexed
    const next = await api.selectVersion(interactionId, k)
    setVers(next)
    setCur(target1) // selection index becomes current
    onResync()
  }

  return (
    <>
      <span className="chev">
        <button
          onClick={() => select(cur - 1)}
          disabled={cur <= 1}
          title="previous version"
        >
          ‹
        </button>
        <span className="ct">
          {cur}&#8202;/&#8202;{total}
        </span>
        <button
          onClick={() => select(cur + 1)}
          disabled={cur >= total}
          title="next version"
        >
          ›
        </button>
      </span>
      <span className="lbl">version</span>
    </>
  )
}
