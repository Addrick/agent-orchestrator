# src/main.py

import asyncio
import os
import sys
import logging

from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.database.context_manager import ContextManager

from src.interfaces.discord_bot import create_discord_bot
from config.global_config import *
from dotenv import load_dotenv
from src.utils.model_utils import get_model_list

load_dotenv('.env')

# Configure logging
LOG_FORMAT = '%(asctime)s - [%(levelname)s] - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s [%(levelname)s][%(name)s:%(lineno)d]: %(message)s',
                    datefmt='[%Y-%m-%d] %H:%M:%S')

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
    # 1. Initialize the database manager, allowing path override via environment variable
    db_path = os.environ.get("DATABASE_FILE")
    context_manager = ContextManager(db_path=db_path)

    # 2. Set up the database schema and default data on startup
    logger.info("Setting up database schema and default data...")
    context_manager.create_schema()
    context_manager._initialize_db()
    logger.info("Database setup complete.")

    # 3. Initialize the centralized text generation engine
    text_engine = TextEngine()

    # 4. Initialize ChatSystem core, injecting dependencies
    bot = ChatSystem(context_manager=context_manager, text_engine=text_engine)

    # 5. Optionally update the model list on startup
    if UPDATE_MODELS_ON_STARTUP:
        logger.info("Updating available models from APIs...")
        # Run the blocking network calls in a separate thread to avoid blocking the event loop
        updated_models = await asyncio.to_thread(get_model_list, update=True)
        if updated_models:
            bot.models_available = updated_models
            logger.info("Model list updated successfully.")
        else:
            logger.error("Failed to update model list.")

    tasks = []

    # --- Initialize Interfaces ---
    logger.info("Starting interface(s)...")

    if DISCORD_BOT:
        logger.info("Initializing Discord bot...")
        discord_bot = create_discord_bot(bot)
        task = asyncio.create_task(discord_bot.start(os.environ.get("DISCORD_API_KEY")))
        tasks.append(task)

    if not DISCORD_BOT:
        print("No standard chat interfaces enabled, defaulting to command line...")
        from src.interfaces.local_terminal import run_terminal_interface
        task = asyncio.create_task(run_terminal_interface(bot))
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
