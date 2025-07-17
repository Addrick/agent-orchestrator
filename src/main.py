import asyncio
import os
import sys
import logging
from datetime import datetime

from src.chat_system import ChatSystem
from src.interfaces.discord_bot import create_discord_bot
from config.global_config import *

import config.api_keys
from dotenv import load_dotenv

from src.chat_system import ChatSystem
from src.interfaces.discord_bot import create_discord_bot
from src.interfaces.gmail_bot import create_gmail_bot  # <-- Unchanged import
from config.global_config import *

from src.utils.model_utils import get_model_list

load_dotenv('.env')

# Configure logging
LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d]: %(message)s'
logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format=LOG_FORMAT,
                    # format='%(asctime)s [%(levelname)s][%(name)s:%(lineno)d]: %(message)s',
                    datefmt='[%Y-%m-%d] %H:%M:%S')

logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('discord').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def main():
    # ... (startup code is the same)

    # Initialize ChatSystem core:
    bot = ChatSystem()
    tasks = []
    # ... (model update task is the same)

    # --- Initialize Interfaces ---
    logger.info(f"Starting interface(s)...")

    if DISCORD_BOT:
        logger.info("Initializing Discord bot...")
        # This pattern is now mirrored below
        discord_bot = create_discord_bot(bot)
        task = asyncio.create_task(discord_bot.start(config.api_keys.discord))
        tasks.append(task)

    if GMAIL_BOT:
        logger.info("Initializing Gmail bot...")
        # Following the same pattern as Discord
        gmail_bot = create_gmail_bot(bot)
        task = asyncio.create_task(gmail_bot.start())
        tasks.append(task)

    # ... (WebUI and other logic is the same)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shutting down.")