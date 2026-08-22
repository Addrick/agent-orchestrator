"""The node's completion ping, turned into a persona turn (DP-343).

An install and a promotion both run detached on the pve node — `systemd-run`
owns them so a multi-GB download survives the SSH disconnect — which means that
when one *finishes*, nothing inside derpr is awake to notice. Before this
module, a finished install sat on the node until a human thought to ask
`install_status` again, and a cold-tier promotion (which `set_active_model`
starts and does not wait for) stopped one step short of serving for the same
reason.

Both node scripts now POST the job id to derpr when a job reaches `done` or
`failed`. This module is what answers that POST:

    ping (job id only) → job_status over SSH → wake the persona → post its reply

Three properties are load-bearing:

1. **The ping carries no facts.** The body is a job id; everything the persona
   is told comes from `HuggingFaceToolHandler.job_status`, i.e. from the same
   SSH read the `install_status` tool uses. A forged or replayed ping therefore
   cannot assert that an install succeeded — at worst it costs one status read
   of a job whose document says exactly what it said before.
2. **The woken turn is filed in the operator's own channel and under the
   operator's own identifier.** Not cosmetic: the persona is CHANNEL_ISOLATED,
   so "activate it when it lands, if I told you to" only works if the woken
   turn can see the channel history where that instruction was given; and a
   write the turn parks is keyed `(user_identifier, persona)` and rendered only
   in the channel whose name matches `ParkedWrite.channel`, so a wake filed
   under a synthetic user would raise approval cards no human can ever see.
3. **Nothing here raises.** A wake failure must not turn into a 500 on the node
   side, where the only consumer is a `curl` in a bash script that has already
   finished the job it was reporting on.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, TYPE_CHECKING

from config import global_config
from src.huggingface.handler import HuggingFaceToolHandler

if TYPE_CHECKING:
    from src.chat_system import ChatSystem
    from src.clients.notification import NotificationRouter

logger = logging.getLogger(__name__)

#: Job ids already woken on, newest last. The node retries its POST twice, and a
#: reply lost on the way back looks exactly like a reply that never came — so a
#: successful wake must be idempotent or a retry re-runs a whole persona turn
#: (and can re-park the same write). Bounded because this is a process-lifetime
#: cache of ids and nothing ever removes one on its own.
_SEEN_LIMIT = 256


class JobCompletionBridge:
    """Wakes one persona when a node job finishes. Wired by the composition root.

    Everything it needs is injected: the handler (the SSH read), the ChatSystem
    (the turn) and the NotificationRouter (the announcement). Tests drive the
    whole path with fakes for all three and no node, no LLM and no Discord.
    """

    def __init__(
        self,
        handler: HuggingFaceToolHandler,
        chat_system: "ChatSystem",
        notification_router: Optional["NotificationRouter"] = None,
    ) -> None:
        self._handler = handler
        self._chat_system = chat_system
        self._notifier = notification_router
        self._seen: "OrderedDict[str, bool]" = OrderedDict()

    # -- entry point ---------------------------------------------------------

    async def handle(self, job_id: str) -> Dict[str, Any]:
        """Answer one completion ping. Returns a small dict for the HTTP route.

        The return value describes what derpr did with the ping — it is a log
        line for the node's journal, not something the node acts on.
        """
        value = str(job_id or "").strip().lower()
        if not value:
            return {"status": "error", "message": "job_id is required"}

        status = await self._handler.job_status(value)
        if status.get("status") != "ok":
            # An unreadable job is a real answer: the node says it finished, the
            # node's own status verb cannot produce the record. Say so and stop —
            # waking a persona with "something finished, I can't tell you what"
            # is worse than the silence this feature replaced.
            logger.warning(
                "job completion ping for %s could not be verified: %s",
                value, status.get("message"),
            )
            return {"status": "error", "message": "job status unreadable",
                    "job_id": value}

        job = status.get("job") or {}
        state = str(job.get("state") or "")
        if state not in ("done", "failed"):
            # The node only pings on a terminal state, so this means the ping
            # overtook the job file or something else sent it. Either way there
            # is nothing to report yet.
            logger.info(
                "job completion ping for %s ignored: state=%r", value, state,
            )
            return {"status": "ignored", "reason": "job not terminal",
                    "job_id": value, "state": state}

        if value in self._seen:
            return {"status": "ignored", "reason": "already handled",
                    "job_id": value}
        self._seen[value] = True
        while len(self._seen) > _SEEN_LIMIT:
            self._seen.popitem(last=False)

        persona = global_config.MODEL_JOB_WAKE_PERSONA
        channel = global_config.MODEL_JOB_WAKE_CHANNEL
        user = global_config.MODEL_JOB_WAKE_USER
        if not (persona and channel and user):
            # Configured off. The status read above still happened, so the log
            # line is the whole feature on an instance that has not set the
            # three values a park needs in order to be approvable.
            logger.info(
                "job %s finished (%s) but no wake is configured "
                "(MODEL_JOB_WAKE_PERSONA/CHANNEL/USER)", value, state,
            )
            return {"status": "ok", "woke": False, "job_id": value,
                    "state": state}

        message = _wake_message(job, status.get("note"))
        try:
            reply, _rtype, _aid, _uid = await self._chat_system.generate_response(
                persona_name=persona,
                user_identifier=user,
                channel=channel,
                message=message,
                user_display_name="model host",
            )
        except Exception:  # noqa: BLE001 — a failed turn must not 500 the node
            logger.exception("wake turn failed for job %s", value)
            return {"status": "error", "message": "wake turn failed",
                    "job_id": value}

        announced = await self._announce(reply)
        return {"status": "ok", "woke": True, "announced": announced,
                "job_id": value, "state": state}

    # -- announcement --------------------------------------------------------

    async def _announce(self, reply: str) -> bool:
        """Post the woken persona's reply to Discord.

        The reply is posted verbatim rather than being handed to the persona as
        a tool it may or may not call. This turn has no human in front of it, so
        an announcement that depends on the model choosing to announce is an
        announcement that goes missing exactly when the install failed and the
        model decided the failure was self-explanatory.
        """
        recipient = global_config.MODEL_JOB_ALERT_CHANNEL_ID
        text = (reply or "").strip()
        if not recipient or not text or self._notifier is None:
            return False
        try:
            return bool(await self._notifier.send(
                channel="discord_channel",
                recipient=recipient,
                subject="",
                body=text,
            ))
        except Exception:  # noqa: BLE001 — best effort, same as the wake itself
            logger.exception("job completion announcement failed")
            return False


def _wake_message(job: Dict[str, Any], note: Optional[str]) -> str:
    """Render one finished job into the user-message text that wakes the persona.

    Every value here came out of `_clean_status`'s whitelist, so this is node
    facts and fixed vocabulary — never an HTTP body, a curl message or anything
    the model typed.
    """
    kind = "promotion" if job.get("kind") == "promote" else "install"
    name = job.get("name") or job.get("file") or job.get("job_id") or "?"
    state = job.get("state")
    lines = [
        f"[model-host] The {kind} of `{name}` finished with state "
        f"`{state}` (job `{job.get('job_id')}`).",
    ]
    if job.get("repo"):
        lines.append(f"Source: {job['repo']}/{job.get('file', '')}")
    elif job.get("file"):
        lines.append(f"File: {job['file']}")
    if job.get("unit"):
        lines.append(f"Unit: {job['unit']}")
    if state == "failed":
        lines.append(f"Failure step `{job.get('step')}`, "
                     f"reason `{job.get('reason')}`.")
    if note:
        lines.append(note)
    lines.append(_INSTRUCTION_FAILED if state == "failed" else _INSTRUCTION_OK[kind])
    return "\n".join(lines)


#: What the woken persona is asked to do. Split by outcome and by kind because
#: the three cases have genuinely different next steps, and a single "decide and
#: act" line left the model to re-derive which of them it was in.
_INSTRUCTION_OK = {
    "install": (
        "Nobody asked you this — the model host reported it. Say what landed, "
        "in one short paragraph. The unit is DISABLED and nothing is serving "
        "it. If this conversation already told you to make it active when it "
        "arrived, do that now: call set_active_model, which will park for "
        "approval, and say that you did. If it did not, do NOT swap :5001 on "
        "your own — report it and offer the swap."
    ),
    "promotion": (
        "Nobody asked you this — the model host reported it. The weights are "
        "now on the SSD, and this did NOT change what :5001 is serving. If "
        "this conversation was working towards making that model active, call "
        "set_active_model again now to finish the swap (it parks for approval) "
        "and say so. Otherwise report that the copy finished and stop."
    ),
}

_INSTRUCTION_FAILED = (
    "Nobody asked you this — the model host reported it. Report the failure "
    "plainly, including the step and reason above, and say what you would do "
    "about it. Do not retry it on your own: a repeat of the same call against "
    "the same cause spends an approval on a known failure."
)
