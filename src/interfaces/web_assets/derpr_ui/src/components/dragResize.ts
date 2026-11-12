// Drag-to-resize from a handle on a panel's LEADING edge (left for width, top
// for height): dragging away from the panel's body grows it. Shared by the
// Inspector width resizer and the LTM dock height (DP-511). Listeners sit on
// window so the drag survives the pointer leaving the thin handle.
export function startDragResize(
  e: React.PointerEvent,
  opts: {
    axis: 'x' | 'y'
    start: number
    min: number
    max: number
    apply: (size: number) => void
    done?: (size: number) => void
  },
): void {
  e.preventDefault()
  const { axis, start, min, max, apply, done } = opts
  const origin = axis === 'x' ? e.clientX : e.clientY
  let size = start
  const cls = axis === 'x' ? ['resizing'] : ['resizing', 'rows']
  document.body.classList.add(...cls)

  const onMove = (ev: PointerEvent) => {
    const pos = axis === 'x' ? ev.clientX : ev.clientY
    size = Math.round(Math.min(max, Math.max(min, start + (origin - pos))))
    apply(size)
  }
  const onUp = () => {
    document.body.classList.remove(...cls)
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onUp)
    done?.(size)
  }
  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', onUp)
}
