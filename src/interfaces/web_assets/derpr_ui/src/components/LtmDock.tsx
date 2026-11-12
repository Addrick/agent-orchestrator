import { useCallback, useRef, useState } from 'react'
import { readPref, writePref } from '../state/persist'
import { startDragResize } from './dragResize'

// DP-511: the recalled LTM block rides on the NEXT user message (DP-431), so
// it previews docked to the composer, outside the transcript scroller. Its
// height changes only when dragged — never when an as-you-type re-fetch
// changes the block's length — so the transcript does not move while typing.
const HEIGHT_PREF = 'ltm_dock_height'
const MIN_HEIGHT = 60
const DEFAULT_HEIGHT = 160
// Leave the transcript at least this much room when the dock is dragged up.
const MIN_TRANSCRIPT = 120

export function LtmDock({ block, personaName }: { block: string; personaName: string }) {
  const [height, setHeight] = useState(() =>
    readPref(HEIGHT_PREF, DEFAULT_HEIGHT, (r) =>
      typeof r === 'number' && Number.isFinite(r) && r >= MIN_HEIGHT ? r : DEFAULT_HEIGHT,
    ),
  )
  const ref = useRef<HTMLDivElement>(null)

  const onResizeStart = useCallback((e: React.PointerEvent) => {
    const dock = ref.current
    // Only the transcript scroller (flex: 1) gives up room to the dock.
    const scroller = dock?.parentElement?.querySelector<HTMLElement>('.scroll')
    if (!dock || !scroller) return
    const start = dock.getBoundingClientRect().height
    startDragResize(e, {
      axis: 'y',
      start,
      min: MIN_HEIGHT,
      max: Math.max(MIN_HEIGHT, start + scroller.clientHeight - MIN_TRANSCRIPT),
      apply: setHeight,
      done: (h) => writePref(HEIGHT_PREF, h),
    })
  }, [])

  return (
    <div className="ltmdock" ref={ref} style={{ height }}>
      <div className="ltmdock-resizer" onPointerDown={onResizeStart} title="drag to resize" />
      <div className="lh">
        <span className="lbl">◈ LTM recalled · prepended to your next message</span>
        <span className="lbl" style={{ marginLeft: 'auto' }}>
          /session/{personaName}/ltm_block
        </span>
      </div>
      <div className="text">{block}</div>
    </div>
  )
}
