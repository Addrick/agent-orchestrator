# src/main.py

import asyncio
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

from src.bootstrap import create_chat_system
from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.memory.memory_manager import MemoryManager
from src.memory.memory_consolidation import MemoryConsolidator
from src.embedding_service import EmbeddingService, GeminiEmbeddingProvider
from src.clients.zammad_client import ZammadClient
from src.clients.zammad_service import ZammadIntegration
from src.app_manager import AppManager
from src.agents.agent_manager import AgentManager
from src.agents.agent_service import AgentServiceIntegration
from src.agents.dispatch_agent import DispatchAgent
from src.agents.managr_agent import ManagrAgent
from src.agents.sqlite_consolidator import SqliteConsolidator
from src.agents.zammad_bot import ZammadBot
from src.agents.reminder_agent import ReminderAgent
from src.agents.date_tagger import DateTagger
from src.agents.content_classifier import ContentClassifier
from src.clients.notification import (
    NotificationRouter,
    DiscordNotifier,
    DiscordChannelNotifier,
    ZammadNotifier
)

from src.security.log_scrub import ScrubbingFormatter
from src.interfaces.discord_bot import create_discord_bot
from src.interfaces.gmail_bot import create_gmail_bot
from src.interfaces.kobold_engine_adapter import create_kobold_engine_adapter
from config.global_config import (
    CHAT_LOG_LOCATION,
    MCP_BRIDGE_ENABLED,
    MCP_BRIDGE_TOOLS,
    LOGS_DIR,
    DISCORD_BOT,
    GMAIL_BOT,
    WEB_INTERFACE,
    KOBOLD_PORT,
    MEMORY_DATABASE_FILE,
    SEMANTIC_BACKEND,
    UPDATE_MODELS_ON_STARTUP,
    DATE_TAGGER_ENABLED,
    DATE_TAGGER_NAME,
    CONTENT_CLASSIFIER_NAME,
)
from dotenv import load_dotenv
from src.utils.model_utils import get_model_list

load_dotenv('.env')


# --- CONFIGURE LOGGING ---
class NoReconnectTracebackFilter(logging.Filter):
    """A custom logging filter to suppress tracebacks for specific reconnect errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Check if the log is from the specific discord.client logger and contains the reconnect message
        if record.name == 'discord.client' and 'Attempting a reconnect' in record.getMessage():
            # If it matches, clear the exception info so no traceback is printed
            record.exc_info = None
            record.exc_text = None
        return True


_LOG_FORMAT = '%(asctime)s [%(levelname)s][%(name)s:%(lineno)d]: %(message)s'
_LOG_DATEFMT = '[%Y-%m-%d] %H:%M:%S'

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format=_LOG_FORMAT,
                    datefmt=_LOG_DATEFMT)

# DP-284: also persist to a rotating file. stdout alone is trapped inside the
# ct100 container's `docker logs`, which a fixr-dispatched agent can't reach —
# so tracebacks (incl. the `[err <id>]` refs surfaced to users) had nowhere a
# reader could tail. LOGS_DIR is a mounted volume; the file bridges that gap.
root_logger = logging.getLogger()
try:
    _file_handler = RotatingFileHandler(
        LOGS_DIR / "derpr.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    root_logger.addHandler(_file_handler)
except OSError as _log_exc:  # never let a bad log path kill startup
    logger_bootstrap = logging.getLogger(__name__)
    logger_bootstrap.warning("Could not open rotating log file: %s", _log_exc)

# DP-284: scrub registered secrets (provider API keys, vault entries) from the
# formatted output — message AND exc_info traceback — before it hits ANY sink.
# derpr.log is permanent + fixr-tailable, so a key in a logged provider error
# must not persist there. Applied to every root handler (stdout + file).
_scrub_formatter = ScrubbingFormatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
for handler in root_logger.handlers:
    handler.setFormatter(_scrub_formatter)
    handler.addFilter(NoReconnectTracebackFilter())

logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('discord').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def update_models_and_sync_bot(bot: ChatSystem) -> None:
    """Fetches the latest model list and updates the live ChatSystem instance."""
    logger.info("Updating available models from APIs...")
    new_models: Optional[Dict[str, Any]] = await asyncio.to_thread(get_model_list, update=True)
    if new_models:
        bot.models_available = new_models
        logger.info("ChatSystem's model list has been synchronized with the latest update.")
    else:
        logger.warning("Failed to fetch new model list; ChatSystem may have stale data.")


def _init_zammad_client() -> Optional[ZammadClient]:
    """Attempt to create a ZammadClient, returning None if credentials are absent."""
    try:
        client = ZammadClient()
        logger.info("Zammad client initialized successfully.")
        return client
    except ValueError:
        logger.warning("Zammad credentials not configured. Zammad features will be disabled.")
        return None


def _register_interfaces(
    app: AppManager,
    bot: ChatSystem,
    notification_router: NotificationRouter,
    date_tagger: Optional[Any] = None,
    mcp_bridge: Optional[Any] = None,
) -> None:
    """Register long-running interface tasks (Discord, Gmail).

    `date_tagger` is the DP-292 LLM date-tagger callable (or None), injected
    into the engine adapter for document-ingest date anchoring.
    `mcp_bridge` is the DP-240 subagent tool bridge (or None), mounted on the
    engine adapter's app.
    """
    if DISCORD_BOT:
        logger.info("Initializing Discord bot...")
        discord_bot = create_discord_bot(bot)
        # DP-230: late-bind the Discord client to the fixr supervisor so agent
        # transcripts/questions can post to per-agent threads. FixrIntegration is
        # constructed before this (step 7.1), so it can't receive the bot at ctor.
        fixr = bot.get_service("fixr")
        if fixr is not None and hasattr(fixr, "attach_discord"):
            fixr.attach_discord(discord_bot)
            # DP-237: once started, surface any agents orphaned by the last
            # restart into their threads (one-shot, awaits Discord readiness).
            if hasattr(fixr, "notify_orphans"):
                app.register_task("fixr_orphan_notify", fixr.notify_orphans())
        # DP-238: late-bind the Discord client to the voice subsystem so it can
        # join its always-listening voice channel (no-op when VOICE_ENABLED off).
        voice = bot.get_service("voice")
        if voice is not None and hasattr(voice, "attach_discord"):
            voice.attach_discord(discord_bot)
        discord_token = os.environ.get("DISCORD_API_KEY")
        if not discord_token:
            logger.error("DISCORD_API_KEY not set. Cannot start Discord bot.")
        else:
            notification_router.register("discord_dm", DiscordNotifier(discord_bot))
            notification_router.register("discord_channel", DiscordChannelNotifier(discord_bot))
            app.register_task("discord", discord_bot.start(discord_token))

    if GMAIL_BOT:
        logger.info("Initializing Gmail bot...")
        gmail_bot = create_gmail_bot(bot)
        app.register_task("gmail", gmail_bot.start())

    if WEB_INTERFACE:
        # Engine-orchestrated Kobold adapter (bespoke DERPR portal at /derpr).
        # The legacy :5002 passthrough adapter was retired in DP-200 (finding A);
        # the engine adapter keeps its established :KOBOLD_PORT+1 (5003) port.
        engine_port = KOBOLD_PORT + 1
        logger.info(f"Initializing Kobold Engine API on port {engine_port}...")
        engine_adapter = create_kobold_engine_adapter(
            bot, date_tagger=date_tagger, mcp_bridge=mcp_bridge)
        engine_adapter.port = engine_port
        # DP-238 web: mount the browser/phone push-to-talk voice capture on the
        # engine adapter's FastAPI app (GET /voice). No-op unless VOICE_WEB_ENABLED.
        voice = bot.get_service("voice")
        if voice is not None and hasattr(voice, "attach_web"):
            voice.attach_web(engine_adapter.app)
        app.register_task("kobold_engine_api", engine_adapter.start())


def _register_agents(
    agent_manager: AgentManager,
    zammad_client: Optional[ZammadClient],
) -> None:
    """Register agent classes with AgentManager. Agents start via auto_start config."""
    # SqliteConsolidator drives the legacy SQLite segment/summary pipeline.
    # Hindsight backend obsoletes it — registering it under the new backend
    # would crash deploy() on first cycle (NotImplementedError on legacy ops).
    if SEMANTIC_BACKEND == "sqlite":
        agent_manager.register("sqlite_consolidator", SqliteConsolidator)
    else:
        logger.info("SqliteConsolidator skipped (SEMANTIC_BACKEND=%s).", SEMANTIC_BACKEND)

    if zammad_client is not None:
        agent_manager.register("zammad_bot", ZammadBot)
        agent_manager.register("dispatch", DispatchAgent)
        agent_manager.register("reminder", ReminderAgent)
        agent_manager.register("managr", ManagrAgent)
    else:
        logger.warning(
            "Zammad credentials missing. Zammad-dependent agents (zammad_bot, dispatch) "
            "will not be registered."
        )

    # Single-shot inference agents (DP-292): DI-built on lookup, never looped.
    # Registered unconditionally.
    #  - date_tagger: the ingest paths use it only when DATE_TAGGER_ENABLED and
    #    regex found no date.
    #  - content_classifier (DP-294): injected into ZammadBot by convention-DI
    #    (its __init__ names `content_classifier`), so it must register before
    #    ZammadBot is built by auto_start.
    agent_manager.register_inference_agent(DATE_TAGGER_NAME, DateTagger)
    agent_manager.register_inference_agent(
        CONTENT_CLASSIFIER_NAME, ContentClassifier)


async def main() -> None:
    """Main asynchronous function to initialize and run the application."""
    logger.info("Starting application...")
    if not os.path.exists(CHAT_LOG_LOCATION):
        os.makedirs(CHAT_LOG_LOCATION)
        logger.warning("Logs folder created!")

    # --- ARCHITECTURE INITIALIZATION ---
    # 1. Initialize the user memory database
    logger.info(f"Initializing database at: {MEMORY_DATABASE_FILE}")
    memory_manager = MemoryManager(db_path=MEMORY_DATABASE_FILE)
    logger.info("Setting up user memory database schema...")
    memory_manager.create_schema()
    logger.info("User memory database setup complete.")

    # 2. Initialize the centralized text generation engine (it owns its
    #    kobold-native local transport — DP-206b facade collapse)
    text_engine = TextEngine()

    # 3. Initialize the Zammad client for ticketing (optional)
    zammad_client = _init_zammad_client()

    # 4. Initialize embedding service for both ChatSystem and background daemons
    embedding_service = EmbeddingService(GeminiEmbeddingProvider())

    # 5. Initialize ChatSystem core, injecting dependencies
    bot = create_chat_system(
        memory_manager=memory_manager,
        text_engine=text_engine,
        embedding_service=embedding_service,
    )

    # 5. Register service integrations
    if zammad_client is not None:
        bot.register_service(ZammadIntegration(zammad_client))

    # 6. Create AgentManager, register agent classes, then register agent tools
    agent_manager = AgentManager(
        chat_system=bot,
        memory_manager=memory_manager,
    )
    _register_agents(agent_manager, zammad_client)
    bot.register_service(AgentServiceIntegration(agent_manager, memory_manager))

    # 7. Create AppManager and NotificationRouter
    app = AppManager(agent_manager=agent_manager)
    notification_router = NotificationRouter()
    agent_manager.notification_router = notification_router
    if zammad_client is not None:
        notification_router.register("zammad", ZammadNotifier(zammad_client))

    # 7.1 Register the fixr self-improvement supervisor (DP-227). Needs both the
    # ChatSystem (the event bridge wakes fixr via generate_response) and the
    # NotificationRouter (fixr's send_discord), so it registers after the router.
    from src.self_edit.integration import FixrIntegration
    bot.register_service(FixrIntegration(bot, notification_router))

    # 7.2 Register the voice command subsystem (DP-238). Needs the
    # NotificationRouter to announce fired timers; the Discord client is
    # late-bound in _register_interfaces (like fixr) so the always-listening
    # capture pipeline can join its voice channel.
    from src.voice import VoiceIntegration
    bot.register_service(VoiceIntegration(notification_router))

    # 7.3 Register the Proxmox management subsystem (DP-262). Registration-only;
    # its tools SSH to the pve node. Registers even when PVE_TOOLS_ENABLED is
    # false so the startup-wiring contract holds (handlers short-circuit disabled
    # calls with an error). Personas opt in via service_bindings: ["proxmox"].
    from src.proxmox import ProxmoxIntegration
    bot.register_service(ProxmoxIntegration())

    # 7.4 Register the MCP client subsystem (DP-268). main.py owns the manager
    # (sessions/lifecycle — voice precedent); the integration only registers
    # the add/remove/list management tools behind the "mcp" binding. Registers
    # even when MCP_ENABLED is false (wiring contract; calls short-circuit).
    from src.tools.mcp_client import MCPClientManager
    from src.tools.mcp_integration import MCPIntegration
    mcp_manager = MCPClientManager(personas_provider=lambda: bot.personas)
    bot.register_service(MCPIntegration(mcp_manager))

    # 7.5 Register the proposal queue review surface (DP-282). Needs Zammad —
    # the executor is the sole component that turns an approved proposal into
    # an external write. Without Zammad there is nothing to execute against,
    # so (like ZammadIntegration) it simply doesn't register.
    # 7.6 DP-240 MCP bridge: build the AgentCallRunner FIRST so the same
    # instance backs both the bridge (which decides what is exposed and what is
    # gated) and the executor (which re-checks that same exposure at approval
    # time). Two instances could drift apart, and the drift would silently widen
    # what an approved row may run.
    mcp_bridge = None
    agent_call_runner = None
    if MCP_BRIDGE_ENABLED:
        from src.proposals.agent_call import AgentCallRunner
        from src.tool_policy import ToolPolicy
        from src.tools.mcp_bridge import McpBridge
        agent_call_runner = AgentCallRunner(
            tool_manager_lookup=lambda: bot.tool_manager,
            policy_lookup=lambda: ToolPolicy(default="deny", allow=list(MCP_BRIDGE_TOOLS)),
        )

        async def _propose_agent_call(agent_id: str, tool_name: str,
                                      tool_args: Dict[str, Any]) -> int:
            return await asyncio.to_thread(
                memory_manager.create_proposal,
                agent_name=f"agent:{agent_id}",
                action_type="call_derpr_tool",
                action_args={"tool_name": tool_name, "tool_args": tool_args,
                             "agent_id": agent_id},
                rationale=f"Dispatched subagent {agent_id} requested {tool_name}.",
            )

        mcp_bridge = McpBridge(agent_call_runner, _propose_agent_call)
        logger.info("MCP bridge enabled; exposing tools: %s", MCP_BRIDGE_TOOLS)

    if zammad_client is not None:
        from src.proposals import ProposalExecutor, ProposalIntegration
        bot.register_service(
            ProposalIntegration(
                memory_manager,
                ProposalExecutor(zammad_client, agent_call_runner=agent_call_runner),
            )
        )
    elif mcp_bridge is not None:
        # Known gap, logged loudly rather than papered over: the approve/deny
        # tools and the executor both live behind ProposalIntegration, which
        # only registers with Zammad present. Without it a subagent can queue a
        # gated call that nothing can ever approve or run.
        logger.warning(
            "MCP bridge is enabled but Zammad is not configured — the proposal "
            "review surface is unregistered, so gated subagent calls will queue "
            "with no way to approve or execute them."
        )

    # 8. Register interfaces
    # DP-292: build the date-tagger callable from AgentManager (convention-DI)
    # and inject it into the engine adapter. Regex-only ingest when disabled.
    date_tagger_callable = None
    if DATE_TAGGER_ENABLED:
        _dt = agent_manager.get_inference_agent(DATE_TAGGER_NAME)
        date_tagger_callable = _dt.tag if _dt is not None else None
    _register_interfaces(app, bot, notification_router, date_tagger=date_tagger_callable,
                         mcp_bridge=mcp_bridge)

    # 8.1 Perform post-init startup tasks (e.g. Hindsight bank provisioning)
    await bot.startup()

    # 8.2 Connect configured MCP servers and register their discovered tools
    # (no-op when MCP_ENABLED is false; a dead server logs and is skipped).
    await mcp_manager.start()

    # 9. Register background daemons — legacy SQLite L1→L2 consolidation only
    if SEMANTIC_BACKEND == "sqlite":
        consolidator = MemoryConsolidator(memory_manager, text_engine, embedding_service)
        app.register_task("memory_consolidator", consolidator.start_daemon(check_interval_seconds=3600))

    # 10. Optionally update the model list on startup
    if UPDATE_MODELS_ON_STARTUP:
        app.register_task("model_update", update_models_and_sync_bot(bot))

    # 11. Run everything (auto_start agents + interface tasks)
    try:
        await app.start()
    finally:
        await mcp_manager.aclose()
        await text_engine.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shutting down.")
