# src/deferral_kinds.py
"""The vocabulary of *deferred tool results* (DP-345).

A **deferral** is one tool call whose real result arrives after the turn that
made it has ended. Nothing hangs: the turn completes with a placeholder tool
result, a durable row binds the pending call to that turn's own coordinates, and
whatever external event finally answers it patches the placeholder and runs one
continuation turn.

`kind` is the axis that names *which* external event does the answering. Its
whole reason to exist is discoverability, so it lives in its own leaf module
rather than beside either consumer:

- `src.tools.tool_loop` writes the placeholder and may not import
  `src.memory.memory_manager` (module-boundary contract);
- `src.memory.memory_manager` stores the row and may not import `src.tools`;
- so a constant both of them need has nowhere else to live that does not
  duplicate it — and a *duplicated* constant whose two copies must stay byte
  equal (the `Parked_Writes.kind` DDL default has to match what the store
  writes) is exactly the failure this ticket was filed about.

The rule a new kind must keep, stated once, here, because it is the rule two
subsystems already broke:

    **The coordinates of a resume come from the turn, never from config.**

`ParkedWrite` carries `user_identifier`, `persona_name`, `channel` and
`server_id`, taken from the turn that raised the deferral. A kind that has to be
*told* which persona or channel to answer in has discarded state it was already
handed. `MODEL_JOB_WAKE_PERSONA`, `MODEL_JOB_WAKE_CHANNEL` and
`MODEL_JOB_WAKE_USER` were exactly that, and they are **deleted** in this change
rather than generalized — the node job's park row already knows all three.
`CC_FIXR_PERSONA` / `CC_FIXR_CHANNEL` are the same defect in fixr and go the
same way when fixr moves onto this mechanism.
"""

from typing import Any, Dict, Optional, Tuple

# A human clicking approve/deny. The original kind. Every kind is addressed by
# TOKEN, including the ones answered by an outside authority: derpr mints the
# park token and hands that same string outward as the job id, so there is no
# second namespace to map back from. Same rule as the coordinates above — a
# kind that has to be handed an identifier the parking call already had is
# discarding state it was given.
#
# Also the value of the `Parked_Writes.kind` DDL default, which is what makes
# the migration a no-op for rows written before the column existed: every one
# of them was an approval.
DEFERRAL_KIND_APPROVAL = "approval"

#: A job running on the pve node — a model install or a cold-tier promotion.
#: Both are detached under `systemd-run` so they outlive the SSH call that
#: started them, and both POST their job id back when they reach a terminal
#: state. That id IS the park token; see the note above.
DEFERRAL_KIND_NODE_JOB = "node_job"

#: A codr ticket stagr queued (DP-493). Answered by the codr agent when the
#: ticket reaches its final outcome (passed, failed with no attempts left, or
#: blocked), so stagr gets one continuation turn per ticket and can queue the
#: next. The ticket ID (`CD-NNN`, never reused) IS the park token.
DEFERRAL_KIND_CODR_TICKET = "codr_ticket"

#: A stagr call that needs codr's box (`codr_shell`, `copy_to_codr`) and found
#: codr busy (DP-517). Answered by the codr agent on the first tick codr is
#: idle, before it starts another ticket, so stagr can run the call again.
#: The token is minted by the refusing handler (`codr-idle-<hex>`); every one
#: pending is settled on that tick, one batch per conversation
#: (`ChatSystem.stream_resolve_deferrals`), so a turn's several refusals get
#: one continuation.
DEFERRAL_KIND_CODR_IDLE = "codr_idle"

#: The kinds whose continuation turn is owed until one completes (DP-518).
#: Settling one of these marks its row `continuation_owed`, and a continuation
#: that errors, is cut off, or dies with the process leaves the mark, so the
#: kind's authority asks again (`ChatSystem.stream_resume_continuations`) until
#: one turn completes — the codr agent, which polls each tick anyway. These are
#: the kinds whose continuation moves the work on: stagr queues the next ticket
#: or reruns the refused call in it, and nobody else would.
#:
#: Never `approval`: an approval's write runs once, in `apply()`, whatever
#: happens to its reply. Not `node_job` either: no authority polls for it after
#: the node's ping (retried once, seconds later), and the outcome is patched into
#: history for the operator's next message, who is the one driving that work.
CONTINUATION_RETRIED_KINDS = frozenset({DEFERRAL_KIND_CODR_TICKET, DEFERRAL_KIND_CODR_IDLE})

#: Key a tool result uses to say "this is not my final answer".
#:
#: A handler that kicks off out-of-band work returns an ordinary result dict
#: (so a caller with no deferral support still reads something sensible) with
#: this one extra key naming the kind that will answer it and the token that
#: answer will arrive under. `declare_deferral` writes it; `declared_deferral`
#: reads it back.
#:
#: Underscore-prefixed because it is plumbing between the handler and the park
#: store, not content: `_settle_deferral` strips it before the result is
#: patched into history, so it never reaches the model.
DEFERRAL_DECLARATION_KEY = "_deferral"


def declare_deferral(result: Dict[str, Any], kind: str,
                     token: str) -> Dict[str, Any]:
    """Mark a tool result as awaiting an external event. Returns `result`.

    Mutates and returns the same dict so a handler can `return
    declare_deferral({...}, KIND, job_id)` in one expression.

    The token must be the identifier the outside world will use to report back
    — for a node job, the `job_id` handed to the node script. It becomes the
    park token verbatim, which is the whole point: nothing has to map one
    namespace onto another later.
    """
    result[DEFERRAL_DECLARATION_KEY] = {"kind": kind, "token": token}
    return result


def declared_deferral(result: Any) -> Optional[Tuple[str, str]]:
    """`(kind, token)` a result declared, or None. Never raises.

    Tolerant on purpose. It reads the return value of an arbitrary tool
    handler, reached through `apply()`, whose job is to not blow up after an
    irreversible write has already run. A malformed declaration means the
    result is treated as final — the same behaviour as a handler that declared
    nothing, which is the pre-DP-345 behaviour and is safe: the work still
    happened, the operator still sees the result, and the only thing lost is
    the automatic resume.
    """
    declared = _declaration(result)
    if not isinstance(declared, dict):
        return None
    kind = declared.get("kind")
    token = declared.get("token")
    if not isinstance(kind, str) or not isinstance(token, str):
        return None
    kind, token = kind.strip(), token.strip()
    if not kind or not token or kind == DEFERRAL_KIND_APPROVAL:
        # An `approval` declaration would mean a tool asking to be re-gated by
        # a human after it already ran. Nothing constructs that, and honouring
        # it would put an approve/deny button on an executed write.
        return None
    return kind, token


def _declaration(result: Any) -> Any:
    """The declaration in a handler's return value, or in the `execute_tool`
    envelope around it (`{"result": <handler's return>}`).

    DP-493: both consumers are handed the ENVELOPE — the approval path's
    `decision.result` and the tool loop's executed result — but this used to
    read only the top level, where a handler's declaration never is. So no
    declared deferral was ever recognized live: an approved `install_model`
    was patched as a completed call and the node's ping found nothing to
    settle. The tests passed because their fake executors return the
    handler's dict unwrapped."""
    if not isinstance(result, dict):
        return None
    if DEFERRAL_DECLARATION_KEY in result:
        return result[DEFERRAL_DECLARATION_KEY]
    inner = result.get("result")
    return inner.get(DEFERRAL_DECLARATION_KEY) if isinstance(inner, dict) else None


def without_declaration(result: Dict[str, Any]) -> Dict[str, Any]:
    """`result` (a handler's return or its envelope) minus the declaration,
    at whichever level it sat: plumbing, never shown to the model."""
    out = {k: v for k, v in result.items() if k != DEFERRAL_DECLARATION_KEY}
    inner = out.get("result")
    if isinstance(inner, dict) and DEFERRAL_DECLARATION_KEY in inner:
        out["result"] = {k: v for k, v in inner.items() if k != DEFERRAL_DECLARATION_KEY}
    return out


__all__ = [
    "DEFERRAL_KIND_APPROVAL",
    "DEFERRAL_KIND_NODE_JOB",
    "DEFERRAL_KIND_CODR_TICKET",
    "DEFERRAL_KIND_CODR_IDLE",
    "CONTINUATION_RETRIED_KINDS",
    "DEFERRAL_DECLARATION_KEY",
    "declare_deferral",
    "declared_deferral",
    "without_declaration",
]
