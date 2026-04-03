---
name: ServiceIntegration lifecycle hooks removed
description: Decision to strip all lifecycle hooks from ServiceIntegration, keeping it as tool-registration-only interface (2026-03-28)
type: project
---

## Decision

Removed all lifecycle methods from `ServiceIntegration` ABC and `ZammadIntegration`:
- `resolve_context`, `on_message`, `prepare_tool_args`, `on_tool_result`, `get_system_messages`, `get_tracking_id`

ServiceIntegration now contains only `name` (abstract property) and `register_tools()`.

## Why

The lifecycle hooks auto-detected tickets from message text, created Zammad users from Discord identifiers, mirrored messages to Zammad tickets, and injected customer_id into tool calls. Adam explicitly does not want this behavior — he manages tickets through direct tool calls ("joy make a ticket for customer@example.com"). The auto-resolution was wasted overhead and risked polluting tickets.

The `service_data` dict that flowed through ChatSystem (built by `resolve_context`, consumed by all other hooks) was the entire pipeline — removing the hooks eliminated service_data entirely.

The `ticket_id` return value from `generate_response()` was unused by all callers (Discord, Gmail, message_handler). Tool call results already contain ticket IDs and are stored in the `tool_context` JSON column.

## What changed

- `ServiceIntegration` ABC: 115 → ~38 lines
- `ZammadIntegration`: 231 → ~37 lines
- `ChatSystem`: removed `_resolve_service_contexts()`, `_notify_services()`, `_get_tracking_id()`, `service_data` from dataclasses
- Return type: `generate_response()` and `resume_pending_confirmation()` changed from 4-tuple to 3-tuple (dropped ticket_id)
- `TICKET_ISOLATED` memory mode: dormant (always returns empty history since ticket_id resolution removed)
- Tests: removed ~25 tests for lifecycle hooks, updated all tuple unpackings

## Commit context

All changes are uncommitted as of 2026-03-28. Plan file: `.claude/plans/jiggly-drifting-gem.md`.
