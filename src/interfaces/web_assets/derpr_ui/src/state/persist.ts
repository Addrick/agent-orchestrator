/* ============================================================
   Client-only UI preferences, persisted to localStorage (DP-273).

   These are pure UI chrome — panel folds, the active inspector tab, the
   per-persona channel — with no server backing, so they reset on every reload
   unless saved here. (Persona samplers/toggles persist server-side via
   PATCH /persona and are re-fetched on boot; they do NOT belong here.)

   Every read runs the stored value through a validator that returns a coerced
   value or the caller's fallback: localStorage is untrusted input (hand-edited,
   corrupt JSON, or a stale enum from an older build), so a bad value must
   degrade gracefully to the default rather than crash the app.
   ============================================================ */

const PREFIX = 'derpr_ui_'

// Read a JSON-encoded pref. `validate` coerces the parsed value to T or returns
// the fallback; any missing key / parse failure also yields the fallback.
export function readPref<T>(
  key: string,
  fallback: T,
  validate: (raw: unknown) => T,
): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    if (raw == null) return fallback
    return validate(JSON.parse(raw))
  } catch {
    return fallback
  }
}

export function writePref(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {
    /* storage full / disabled (private mode) — a dropped pref is harmless */
  }
}
