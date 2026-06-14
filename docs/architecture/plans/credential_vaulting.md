---
name: Credential scoping (vault + egress scrubber)
description: Central credential vault plus a fail-safe egress scrubber that redacts machine secrets from any string bound for the LLM context, audit log, or inspector. DP-225.
type: project
status: in_progress
---

# Credential Scoping — DP-225

Supersedes the prior deferred "Credential Vaulting" sketch. Goal: machine secrets
never leak into the LLM context window (or audit records / the `/assemble`
inspector). Adapted from Avibe's vault concept (their vault is roadmap-only, so
this is an original implementation).

## Threat model

Secrets in scope: provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_GENERATIVEAI_API_KEY`), `ZAMMAD_API_KEY`, and any future credential.

Leakage boundaries (everything the model can read back):
1. **Tool results** — `src/tools/tool_loop.py` `json.dumps(tool_result)` appended
   to `conversation_history` and re-sent each iteration. Primary vector.
2. **Audit args** — `audit_actions[].arguments` persisted to `Agent_Actions` and
   re-rendered in the confirmation message.
3. **Cached `api_payload`** (`last_api_requests`) — surfaced by `/assemble` and the
   portal inspector.

Today the curated tool surface means no tool returns secrets and the model never
receives them as args. This is defense-in-depth: the guarantee must already hold
the moment a shell/exec tool or BYO-credential mode lands.

## Design — two pillars, fail-safe

### A. Central vault — `src/security/vault.py`
`CredentialVault`: the single inventory of machine secrets.
- `get(ref) -> str | None` and `require(ref) -> str` read from env (pluggable
  source; encrypted-file/keyring can be added behind the same interface later).
- `known_refs()` enumerates the secret keys it manages.
- On construction (or `register_into(scrubber)`), every resolved secret VALUE is
  registered with the egress scrubber.
- Replaces scattered `os.environ.get("...API_KEY")` calls in `engine.py` and
  `zammad_client.py` so there is exactly one place that knows what the secrets are.

### B. Egress scrubber — `src/security/scrubber.py`
Process-global `SecretScrubber`, accessed via `get_scrubber()` (module singleton;
secrets are a process-wide property, so threading it through every constructor is
unnecessary and error-prone — tests reset/populate it explicitly).
- `register(value, ref)` adds a secret value→label.
- `scrub(obj)` walks str / dict / list and replaces each registered value with
  `[REDACTED:<ref>]`. Longest-value-first replacement so overlapping secrets don't
  partially leak.
- Pattern fallback redacts common secret shapes (`sk-[A-Za-z0-9]{20,}`,
  `Token token=...`, bearer tokens) to catch *unregistered* leaks → `[REDACTED:pattern]`.
- Min-length guard (≥8 chars) so short/empty values are ignored (avoids redacting
  benign substrings).

### Enforcement points
| Boundary | File | Action |
|----------|------|--------|
| Tool result → context | `src/tools/tool_loop.py` `_execute_calls` | `scrub(result_str)` before append + emit |
| Audit args → DB/confirmation | `src/tools/tool_loop.py` write-park block | `scrub(wc_args)` into `audit_actions` |
| Payload cache → inspector | wherever `last_api_requests` is set | `scrub(payload)` before caching |

## Sprints
- **S1** foundation: `src/security/` (vault + scrubber) + unit tests. No wiring.
- **S2** egress enforcement: wire scrubber at the three boundaries + boundary tests.
- **S3** vault adoption: route provider/zammad secrets through the vault; bootstrap
  builds the vault and registers secrets into the scrubber; startup wiring test.
- **S4** docs + memory; optional encrypted-at-rest vault file.

## Non-goals (this DP)
- Encrypted-at-rest vault file / keyring backend (interface leaves room; not built unless trivial in S4).
- Per-persona credential scoping / multi-tenant key isolation (revisit when multi-tenant lands).
