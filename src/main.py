# src/main.py

import asyncio
import os
import sys
import logging
from typing import Optional, Dict, Any

from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.stream_engine import StreamEngine
from memory.memory_manager import MemoryManager
from memory.memory_consolidation import MemoryConsolidator
from src.embedding_service import EmbeddingService, GeminiEmbeddingProvider
from src.clients.zammad_client import ZammadClient
from src.clients.zammad_service import ZammadIntegration
from src.app_manager import AppManager
from src.agents.agent_manager import AgentManager
from src.agents.agent_service import AgentServiceIntegration
from src.agents.dispatch_agent import DispatchAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.zammad_bot import ZammadBot
from src.agents.reminder_agent import ReminderAgent
from src.clients.notification import (
    NotificationRouter,
    DiscordNotifier,
    DiscordChannelNotifier,
    ZammadNotifier
)

from src.interfaces.discord_bot import create_discord_bot
from src.interfaces.gmail_bot import create_gmail_bot
from src.interfaces.kobold_adapter import create_kobold_adapter
from config.global_config import (
    CHAT_LOG_LOCATION,
    DISCORD_BOT,
    GMAIL_BOT,
    WEB_INTERFACE,
    KOBOLD_PORT,
    MEMORY_DATABASE_FILE,
    UPDATE_MODELS_ON_STARTUP,
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


logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s [%(levelname)s][%(name)s:%(lineno)d]: %(message)s',
                    datefmt='[%Y-%m-%d] %H:%M:%S')

root_logger = logging.getLogger()
for handler in root_logger.handlers:
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
) -> None:
    """Register long-running interface tasks (Discord, Gmail)."""
    if DISCORD_BOT:
        logger.info("Initializing Discord bot...")
        discord_bot = create_discord_bot(bot)
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
        logger.info(f"Initializing Kobold Web API on port {KOBOLD_PORT}...")
        kobold_adapter = create_kobold_adapter(bot)
        # Configure the adapter's port from the config
        kobold_adapter.port = KOBOLD_PORT
        app.register_task("kobold_api", kobold_adapter.start())


def _register_agents(
    agent_manager: AgentManager,
    zammad_client: Optional[ZammadClient],
) -> None:
    """Register agent classes with AgentManager. Agents start via auto_start config."""
    # Memory agent — no external service dependency
    agent_manager.register("memory", MemoryAgent)

    if zammad_client is not None:
        agent_manager.register("zammad_bot", ZammadBot)
        agent_manager.register("dispatch", DispatchAgent)
        agent_manager.register("reminder", ReminderAgent)
    else:
        logger.warning(
            "Zammad credentials missing. Zammad-dependent agents (zammad_bot, dispatch) "
            "will not be registered."
        )


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

    # 2. Initialize the centralized text generation engine
    stream_engine = StreamEngine()
    text_engine = TextEngine(stream_engine=stream_engine)

    # 3. Initialize the Zammad client for ticketing (optional)
    zammad_client = _init_zammad_client()

    # 4. Initialize embedding service for both ChatSystem and background daemons
    embedding_service = EmbeddingService(GeminiEmbeddingProvider())

    # 5. Initialize ChatSystem core, injecting dependencies
    bot = ChatSystem(
        memory_manager=memory_manager,
        text_engine=text_engine,
        embedding_service=embedding_service,
        stream_engine=stream_engine,
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

    # 8. Register interfaces
    _register_interfaces(app, bot, notification_router)

    # 9. Register background daemons
    consolidator = MemoryConsolidator(memory_manager, text_engine, embedding_service)
    app.register_task("memory_consolidator", consolidator.start_daemon(check_interval_seconds=3600))

    # 10. Optionally update the model list on startup
    if UPDATE_MODELS_ON_STARTUP:
        app.register_task("model_update", update_models_and_sync_bot(bot))

    # 11. Run everything (auto_start agents + interface tasks)
    try:
        await app.start()
    finally:
        await stream_engine.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shutting down.")
