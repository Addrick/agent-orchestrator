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

from src.utils.model_utils import get_model_list

load_dotenv('.env')

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - [%(levelname)s] - [%(name)s] -  %(message)s',
                    datefmt='[%Y-%m-%d] %H:%M:%S')
logger = logging.getLogger(__name__)


async def run_terminal_interface(bot: ChatSystem):
    """
    Asynchronous function to run the command-line interface.
    """
    from src.interfaces import local_terminal
    client = local_terminal.Client()
    logger.info("Command-line interface is ready.")
    while True:
        # Run the blocking input() in a separate thread to avoid blocking the asyncio event loop
        message_content = await asyncio.to_thread(input, "Enter a message: ")

        if message_content.strip():
            # Create a simulated message object
            current_time = datetime.now().time()
            simulated_message = local_terminal.StrippedMessage(
                message_content,
                author=local_terminal.User(),
                channel=local_terminal.Channel(),
                guild=local_terminal.Guild(),
                timestamp=current_time
            )
            # Process the message asynchronously
            await local_terminal.on_message(bot, simulated_message)


async def main():
    """
    Main asynchronous function to initialize and run all interfaces.
    """
    print("Starting derpr...")
    if not os.path.exists(CHAT_LOG_LOCATION):
        os.makedirs(CHAT_LOG_LOCATION)
        logger.warning("Logs folder created!")

    # Initialize ChatSystem core:
    bot = ChatSystem()

    # A list to hold all the tasks we want to run concurrently
    tasks = []

    # --- Initialize Interfaces ---

    if DISCORD_BOT:
        logger.info("Initializing Discord bot...")
        discord_bot = create_discord_bot(bot)
        task = asyncio.create_task(discord_bot.start(config.api_keys.discord))
        tasks.append(task)

    if WEBUI:  # TODO: find solution for easy web chat; not a fan of gradio
        logger.info("Initializing WebUI...")
        from src.basic_webui import WebUI
        tasks.append(asyncio.create_task(WebUI.launch()))

    # --- Default to command line if no other standard interface is enabled ---
    if not DISCORD_BOT and not WEBUI:
        print("No standard chat interfaces enabled, defaulting to command line...")
        task = asyncio.create_task(run_terminal_interface(bot))
        tasks.append(task)

    # --- Startup Tasks ---
    UPDATE_ON_STARTUP = True
    if UPDATE_ON_STARTUP:
        update_task = asyncio.create_task(asyncio.to_thread(get_model_list, update=True))
        tasks.append(update_task)

    # --- Run all tasks concurrently ---
    if not tasks:
        logger.warning("No interfaces were enabled or created. The application will now exit.")
        return

    logger.info(f"Starting {len(tasks)} interface(s)...")
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shutting down.")