"""ServiceIntegration for `notify_adam`, the operator notification tool (DP-512).

A persona with the `notify` binding and `notify_adam` in its tools can message
Adam from inside a turn. Where it goes is config, never an argument:

    agents.json
      "notify": {
        "per_hour": 20,
        "targets": {
          "default": {"channel": "discord_dm", "recipient": "adrich"},
          "fixr":    {"channel": "discord_channel", "recipient": "debug"}
        }
      }

A target has the shape of an agent's `notification_targets` entry and its
recipient is resolved through the same `recipients` block. The post and the
stored reply go through `clients.notification.post_as_persona`, the path
codr's opencode relay uses too.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Tuple

from src.clients.notification import HourlyLimiter, post_as_persona, resolve_recipient
from src.clients.service_integration import ServiceIntegration
from src.tools.turn_context import get_turn_context

if TYPE_CHECKING:
    from src.clients.notification import NotificationRouter
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

DEFAULT_PER_HOUR = 20
DEFAULT_TARGET_KEY = "default"


def _not_sent(reason: str) -> Dict[str, Any]:
    return {"status": "not_sent", "message": f"Not sent: {reason}"}


class NotifyIntegration(ServiceIntegration):
    def __init__(
        self,
        notification_router: "NotificationRouter",
        persistence: Any,
        config_provider: Callable[[], Mapping[str, Any]],
    ) -> None:
        """`persistence` is the `TurnPersistence`; `config_provider` returns
        the loaded agents.json, read on every call so an edit needs no
        re-wiring."""
        self._router = notification_router
        self._persistence = persistence
        self._config = config_provider
        self._limiters: Dict[str, HourlyLimiter] = {}

    @property
    def name(self) -> str:
        return "notify"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        tool_manager.register("notify_adam", self._notify_adam)

    def _settings(self) -> Mapping[str, Any]:
        settings = (self._config() or {}).get("notify")
        return settings if isinstance(settings, Mapping) else {}

    def _target(self, persona_name: str) -> Optional[Tuple[str, str]]:
        """(router channel, resolved recipient id) for a persona: its own
        entry under `notify.targets`, else `default`; None when neither
        resolves."""
        targets = self._settings().get("targets")
        if not isinstance(targets, Mapping):
            return None
        target = targets.get(persona_name) or targets.get(DEFAULT_TARGET_KEY)
        if not isinstance(target, Mapping):
            return None
        channel = str(target.get("channel") or "discord_dm")
        recipient = resolve_recipient(
            (self._config() or {}).get("recipients"), channel, target.get("recipient"))
        return (channel, recipient) if recipient else None

    def _limiter(self, persona_name: str) -> HourlyLimiter:
        try:
            per_hour = int(self._settings().get("per_hour", DEFAULT_PER_HOUR))
        except (TypeError, ValueError):
            per_hour = DEFAULT_PER_HOUR
        limiter = self._limiters.get(persona_name)
        if limiter is None or limiter.per_hour != per_hour:
            limiter = self._limiters[persona_name] = HourlyLimiter(per_hour)
        return limiter

    async def _notify_adam(self, message: Any = None) -> Dict[str, Any]:
        ctx = get_turn_context()
        if ctx is None:
            return _not_sent("this call is not part of a persona's turn.")
        text = message.strip() if isinstance(message, str) else ""
        if not text:
            return _not_sent('notify_adam needs a non-empty "message" string. Send it again.')
        target = self._target(ctx.persona_name)
        if target is None:
            return _not_sent(
                f"no notify target is configured for '{ctx.persona_name}' "
                "(agents.json notify.targets). Say what you wanted to tell Adam in your reply.")
        channel, recipient = target
        if channel not in self._router.available_channels:
            return _not_sent(f"the '{channel}' channel is not connected. Say it in your reply instead.")
        if await self._router.conversation_key(channel, recipient) == (ctx.channel, ctx.server_id):
            return _not_sent("you are already talking to Adam here. Answer in your reply.")
        if not self._limiter(ctx.persona_name).allow():
            logger.info("notify_adam rate limit hit for '%s'; dropped: %s", ctx.persona_name, text[:200])
            return _not_sent("you have reached this hour's message limit. Keep it for your reply.")
        sent = await post_as_persona(
            self._router, self._persistence, persona_name=ctx.persona_name,
            user_identifier=f"agent:{ctx.persona_name}", channel=channel, recipient=recipient,
            body=f"{ctx.persona_name}: {text}", reply_text=text)
        if not sent:
            return _not_sent("the post failed. Say it in your reply instead.")
        return {"status": "sent", "message": "Sent to Adam. Keep working."}
