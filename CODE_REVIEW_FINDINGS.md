# DP-132 bespoke-UI branch — code review findings

`/code-review xhigh` against merge-base `2b326bb` … tip `96f5a26`. 13 findings
verified by reading the cited code. This file is the handoff artifact: a fresh
session can reconstruct full context from the finding + the cited functions.

## Status summary

| # | Area | File | Status |
|---|------|------|--------|
| 1 | edit erases reasoning | MessageRow.tsx / memory_manager.py | **FILED → DP-141** (preexisting) |
| 2 | read mutates hello-window | chat_system.py / persona.py | **FILED → DP-142** (preexisting) |
| 3 | stale tool_context on regen | memory_manager.py | **FIXED** (`_UNSET` sentinel + test) |
| 4 | onError never clears `active` | store.ts | **FIXED** |
| 5 | _parse_tool_context AttributeError → 500 | kobold_export.py | **FIXED** |
| 6 | onDone re-syncs wrong channel | store.ts | **FIXED** |
| 7 | duplicate-content canonical multi-flag | memory_manager.py | **FIXED** (retry dedupe + test) |
| 8 | get_distinct_channels split by server_id | memory_manager.py | **FIXED** (group by channel + test) |
| 9 | LTM row positional index===1 | Conversation.tsx | **FIXED** (first-user-chunk by identity) |
| 10 | toggleLtm no rollback | store.ts | **FIXED** |
| 11 | patchPersona ignores r.ok | client.ts / store.ts | **FIXED** |
| 12 | /assemble parity overstated | kobold_engine_adapter.py | **FIXED** (claim scoped to portal client) |
| 13 | _parse_tool_context raw fallthrough | kobold_export.py | **FILED → DP-143** (preexisting) |

## New-bug fix wave (#3, #7, #8, #9, #12) — this session

The five findings INTRODUCED by the DP-132–139 work are now fixed on this branch:
- **#3** `update_interaction_content` uses an `_UNSET` sentinel for `tool_context`
  (omitted = leave / `None` = clear / value = set), so a regen to a plain-text
  answer clears stale tool_context instead of rendering phantom tool cards. The
  regen caller (chat_system.py:950) passes `None` to clear; the PATCH manual-edit
  caller omits it. Test: `test_update_interaction_content_tool_context_sentinel`.
- **#7** `handle_portal_retry` content-hash-dedupes the archive insert (mirrors
  `swap_interaction_version`), so identical regens don't create duplicate archive
  rows and only one version is flagged canonical → chevron findIndex is correct.
  Test: `test_handle_portal_retry_dedupes_identical_content`.
- **#8** `get_distinct_channels` groups by channel only (`MAX(server_id)` as a
  representative), so a channel logged under both NULL and a non-NULL server_id
  renders once with combined count. Test:
  `test_get_distinct_channels_groups_by_channel_only`.
- **#9** Conversation.tsx places the LTM author's-note row after the FIRST real
  user chunk located by identity (was positional `index === 1`, which vanished on
  a single-chunk transcript and mispositioned when chunk[0] was an assistant
  turn). `RenderedSlot` now takes a `showLtm` boolean instead of `index`.
- **#12** `/assemble` docstring + RawPane banner now SCOPE the parity guarantee
  to the bespoke portal client (`user=portal`, persona-default samplers, no
  server_id — the only caller). Broadening /assemble to accept arbitrary
  user/server/samplers is explicitly deferred (the claim is now true as-stated).

Gates: `pytest tests/memory tests/interfaces/test_kobold_engine_adapter.py
tests/test_chat_system.py` green (257 passed); mypy clean on the 3 changed
backend files; UI eslint + `tsc -b` + vite build green; flake8 added no new
violations.

The three preexisting findings (#1, #2, #13) are filed as DP-141 / DP-142 /
DP-143 in `memory/project/tasks/` with root-cause, fix options, and mandatory
tests, to be fixed after this wave.

## Fixed this session (#4, #5, #6, #10, #11)

- **#4** `store.ts` `onError` now sets `active:false` (only `onDone` did, and the
  HTTP-non-OK / fetch-reject paths bypass `onDone`, leaving the Composer stuck
  disabled showing "■ stop").
- **#5** `kobold_export.py` `_parse_tool_context` skips non-dict list elements /
  non-dict tool_calls and broadens the except to include `AttributeError`, so a
  malformed `tool_context` no longer 500s the whole `/transcript`.
- **#6** `store.ts` `refreshTranscript` takes an optional `channel`; `buildHandlers`
  captures the channel the turn was sent under and `onDone` re-syncs THAT channel
  (not `activeChannelRef.current`), so a mid-stream channel switch no longer drops
  the completed turn from view.
- **#10** `store.ts` `toggleLtm` awaits the PATCH and rolls back the optimistic
  `ltmOn`/`ltmBlock` flip on failure (+ banner).
- **#11** `client.ts` `patchPersona` now throws on `!r.ok`; both callers
  (`toggleLtm`, `savePersona`) handle the throw (savePersona returns an `error`
  result + banner instead of an unhandled rejection).

## Deferred — need a decision and/or mandatory tests

### #1 — Editing an assistant message erases its `<think>` reasoning  (DATA LOSS)
`MessageRow.saveEdit` (MessageRow.tsx:53) sends only `splitThink(content).body`;
`update_interaction_content` (memory_manager.py:619-624) writes
`reasoning_content = NULL` when not passed (PATCH path passes only content).
**Decision:** either (a) the UI sends the full `<think>…</think>` content and the
server re-splits, or (b) `update_interaction_content` preserves `reasoning_content`
when `None` (same "don't touch" semantics it already uses for `tool_context`).
Prefer (b) for symmetry. Needs a memory_manager unit test.

### #2 — Read paths advance the 'hello' temp_history_override
`get_view_history` (chat_system.py:390) and `assemble_request`
(via `_build_conversation_history`:414) call `persona.get_history_messages()`,
which mutates `_temp_history_override += 2` (persona.py:165). The transcript is
re-synced after every turn/load → with an active hello window the counter inflates
spuriously. **Fix:** give the view/dry-run paths a non-mutating read (peek the
limit without advancing); the LIVE turn path must keep advancing. Needs a test
asserting a transcript fetch does not change the override.

### #3 — Regen to a no-tool answer keeps stale tool_context
`update_interaction_content`'s `None` means "don't touch", but a regen that
produced no tools passes `tool_context=None` (chat_system.py:950) → old
tool_context survives → phantom tool cards. **Fix:** use a sentinel that
distinguishes "clear" from "leave" (e.g. a default-`_UNSET` marker, or pass `""`
to clear). Needs a test. Note: interacts with #1 (same method's None-handling).

### #7 — Duplicate-content versions get multiply-flagged canonical
`list_interaction_versions` flags `canonical=True` on every row whose content ==
canonical (memory_manager.py:677); `handle_portal_retry` (line 568) inserts
archives with no dedupe, so identical-content archives are possible, and the
chevron's `findIndex(canonical)` then picks the wrong index. **Fix:** flag by
`edit_id`/identity rather than content equality, or dedupe archives. The swap path
already content-dedupes (lines 752-761); align the retry insert. Needs a test.

### #8 — get_distinct_channels splits one channel across server_ids
`GROUP BY channel, server_id` (memory_manager.py:996) → a channel logged under
both NULL and non-NULL server_id shows twice in the rail. **Decision:** group by
channel only (and pick a representative/`MAX` server_id), or keep per-server but
de-dupe in the UI. Read-only query; needs a unit test for the chosen semantics.

### #9 — LTM row placement uses positional `index === 1` (Conversation.tsx:216)
Cosmetic: the "◈ LTM recalled" row mispositions / vanishes when chunk[1] isn't the
first assistant turn. Place it after the first non-ephemeral user chunk by id.

### #12 — /assemble parity claim is overstated (kobold_engine_adapter.py:317)
`/assemble` hardcodes `user_identifier="portal"`, ignores `server_id` and sampler
overrides; the docstring/banner claim unconditional parity. Holds for the bespoke
UI (it only sends persona-default samplers + default user), but a PERSONAL/SERVER
persona driven by a different user, or a kobold-lite submit, would diverge.
**Fix:** accept those params on /assemble, or soften the docstring/banner to scope
the claim to the portal.

### #13 — _parse_tool_context returns raw OpenAI msgs when no tool_calls resolve
Returning the raw list (kobold_export.py:182) hands the frontend a `{role,content}`
shape instead of `ToolContext[]`. **BUT** existing tests
(`test_parse_tool_context_passthrough_already_structured`,
`test_parse_tool_context_non_list_returned_as_is`) encode raw passthrough as
intended. So the real fix must distinguish "already-structured ToolContext list"
(no `role` key → pass through) from "raw OpenAI messages that failed to resolve"
(have `role` → return None). Behavioral change + test updates → deferred.

## Refuted candidates (checked, NOT bugs — don't re-flag)

- **Chevron `setCur(target1)` desync after swap** — REFUTED. `swap_interaction_version`
  keeps a stable archive order with content-hash dedupe (memory_manager.py:752-761),
  so multi-click nav stays consistent (matches the recent fix). Only the
  duplicate-content edge (#7) remains.
- **`regen` sends `' '` as user text** — REFUTED. Engine guards
  `sidecar_user.strip()` (kobold_engine_adapter.py:859) and falls back to DB retry.
- **`getChannels` hardcoded `active: c.channel==='web_ui'`** — REFUTED (dead field).
  Channels.tsx highlights via `it.channel === activeChannel`.
- **client_messages / image_url parity** — REFUTED for the streaming path
  (`client_messages=None` forced at kobold_engine_adapter.py:1000; UI sends no
  images).
- **empty-content assistant row persisted** — by design (tool-only turns).
