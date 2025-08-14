# src/main.py

import asyncio
import os
import sys
import logging

from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.database.memory_manager import MemoryManager
from src.clients.zammad_client import ZammadClient

from src.interfaces.discord_bot import create_discord_bot
from src.interfaces.gmail_bot import create_gmail_bot
from config.global_config import *
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
            record.exc_info = 'Discord disconnected, attempting to reconnect...'
            record.exc_text = None
        return True


LOG_FORMAT = '%(asctime)s - [%(levelname)s] - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
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


async def main():
    """Main asynchronous function to initialize and run the application."""
    print("Starting application...")
    if not os.path.exists(CHAT_LOG_LOCATION):
        os.makedirs(CHAT_LOG_LOCATION)
        logger.warning("Logs folder created!")

    # --- ARCHITECTURE INITIALIZATION ---
    # 1. Initialize the user memory database
    memory_db_path = os.environ.get("MEMORY_DATABASE_FILE")
    memory_manager = MemoryManager(db_path=memory_db_path)
    logger.info("Setting up user memory database schema...")
    memory_manager.create_schema()
    logger.info("User memory database setup complete.")

    # 2. Initialize the centralized text generation engine
    text_engine = TextEngine()

    # 3. Initialize the Zammad client for ticketing
    zammad_client = ZammadClient()

    # 4. Initialize ChatSystem core, injecting dependencies
    bot = ChatSystem(
        memory_manager=memory_manager,
        text_engine=text_engine,
        zammad_client=zammad_client
    )


    tasks = []

    # --- Initialize Interfaces ---
    logger.info("Starting interface(s)...")

    if DISCORD_BOT:
        logger.info("Initializing Discord bot...")
        discord_bot = create_discord_bot(bot)
        task = asyncio.create_task(discord_bot.start(os.environ.get("DISCORD_API_KEY")))
        tasks.append(task)

    if GMAIL_BOT:
        logger.info("Initializing Gmail bot...")
        gmail_bot = create_gmail_bot(bot)
        task = asyncio.create_task(gmail_bot.start())
        tasks.append(task)

    if not tasks:
        logger.info("Initializing local terminal interface...")
        from src.interfaces.local_terminal import run_terminal_interface
        task = asyncio.create_task(run_terminal_interface(bot))
        tasks.append(task)

    # 5. Optionally update the model list on startup
    if UPDATE_MODELS_ON_STARTUP:
        logger.info("Updating available models from APIs...")
        # Run the blocking network calls in a separate thread to avoid blocking the event loop
        task = asyncio.to_thread(get_model_list, update=True)
        tasks.append(task)

    if not tasks:
        logger.warning("No interfaces were enabled. The application will exit.")
        return

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shutting down.")
