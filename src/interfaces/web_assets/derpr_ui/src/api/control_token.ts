/* ============================================================
   Operator control token (DP-277).

   The portal's control plane (persona create/edit, dev commands, confirm,
   interaction edits) is gated server-side by DERPR_CONTROL_TOKEN. The
   operator pastes the token once; we keep it in localStorage and attach it
   as `Authorization: Bearer <token>` on every mutating request (and on chat,
   so typed dev commands elevate).

   This is the operator's OWN credential entered into their OWN browser — it
   is never surfaced to the model, persona, or any tool result. Do not log it.
   ============================================================ */

const KEY = 'derpr_control_token'

let _token = ''
try {
  _token = localStorage.getItem(KEY) || ''
} catch {
  /* localStorage unavailable (private mode / SSR build) — memory-only */
}

export function getControlToken(): string {
  return _token
}

export function setControlToken(token: string): void {
  _token = token.trim()
  try {
    if (_token) localStorage.setItem(KEY, _token)
    else localStorage.removeItem(KEY)
  } catch {
    /* memory-only fallback */
  }
}

export function hasControlToken(): boolean {
  return _token.length > 0
}

/** Merge the operator auth header into a fetch headers object when a token
 *  is set. No token → headers unchanged (request goes out anonymous and the
 *  server answers 401 for control-plane routes). */
export function withAuth(headers: Record<string, string> = {}): Record<string, string> {
  return _token ? { ...headers, Authorization: `Bearer ${_token}` } : headers
}
