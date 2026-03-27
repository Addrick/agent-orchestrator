# src/main.py

import asyncio
import os
import sys
import logging
from typing import Optional, Dict, Any

from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.database.memory_manager import MemoryManager
from src.clients.zammad_client import ZammadClient
from src.clients.zammad_service import ZammadIntegration
from src.app_manager import AppManager
from src.agents.agent_manager import AgentManager
from src.agents.agent_service import AgentServiceIntegration
from src.agents.dispatch_agent import DispatchAgent
from src.agents.zammad_bot import ZammadBot
from src.clients.notification import NotificationRouter, DiscordNotifier, ZammadNotifier

from src.interfaces.discord_bot import create_discord_bot
from src.interfaces.gmail_bot import create_gmail_bot
from config.global_config import (
    CHAT_LOG_LOCATION,
    DISCORD_BOT,
    GMAIL_BOT,
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
            notification_router.register("discord", DiscordNotifier(discord_bot))
            app.register_task("discord", discord_bot.start(discord_token))

    if GMAIL_BOT:
        logger.info("Initializing Gmail bot...")
        gmail_bot = create_gmail_bot(bot)
        app.register_task("gmail", gmail_bot.start())


def _register_agents(
    agent_manager: AgentManager,
    zammad_client: Optional[ZammadClient],
) -> None:
    """Register agent classes with AgentManager. Agents start via auto_start config."""
    if zammad_client is not None:
        agent_manager.register("zammad_bot", ZammadBot)
        agent_manager.register("dispatch", DispatchAgent)
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
    text_engine = TextEngine()

    # 3. Initialize the Zammad client for ticketing (optional)
    zammad_client = _init_zammad_client()

    # 4. Initialize ChatSystem core, injecting dependencies
    bot = ChatSystem(
        memory_manager=memory_manager,
        text_engine=text_engine,
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

    # 9. Optionally update the model list on startup
    if UPDATE_MODELS_ON_STARTUP:
        app.register_task("model_update", update_models_and_sync_bot(bot))

    # 10. Run everything (auto_start agents + interface tasks)
    await app.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shutting down.")
