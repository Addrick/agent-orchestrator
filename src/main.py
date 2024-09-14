import asyncio
import os
import sys

from src import local_terminal
from src.chat_system import ChatSystem
from src.discord_bot import *
from config.global_config import *

import config.api_keys

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(message)s',
                    datefmt='[%Y-%m-%d] %H:%M:%S')
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    if not os.path.exists(CHAT_LOG_LOCATION):
        os.makedirs(CHAT_LOG_LOCATION)
        logger.warning("Logs folder created!")

    # Initialize ChatSystem core:
    bot = ChatSystem()

    # Initialize bots:
    if DISCORD_BOT:
        discord_bot = create_discord_bot(bot)
        discord_bot.run(config.api_keys.discord)

    else:
        from datetime import datetime

        client = local_terminal.Client()
        while 1:
            message = input("Enter a message: ")
            # Create a simulated message object
            current_time = datetime.now().time()
            simulated_message = local_terminal.StrippedMessage(message,
                                                               author=local_terminal.User(),
                                                               channel=local_terminal.Channel(),
                                                               guild=local_terminal.Guild(),
                                                               timestamp=current_time)

            # Process the message using local_terminal
            asyncio.create_task(local_terminal.on_message(bot, simulated_message))
