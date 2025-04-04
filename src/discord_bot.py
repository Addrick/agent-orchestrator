import logging
import re

import discord
from discord import HTTPException

from config import global_config
from config.global_config import *
from config.global_config import DISCORD_CHAR_LIMIT
from src.utils.message_utils import split_string_by_limit


# Summary:
# This code implements a Discord bot using the discord.py library. The bot manages multiple personas,
# responds to messages, and handles various commands. It includes features like logging, context
# gathering, and dynamic status updates. The bot uses an external ChatSystem for generating responses.

class ConnectionErrorFilter(logging.Filter):
    def filter(self, record):
        # Filter out specific connection-related log messages
        connection_keywords = [
            'Attempting a reconnect',
            'WebSocket closed',
            'ConnectionClosed',
            'ClientConnectorError'
        ]
        return not any(keyword in record.getMessage() for keyword in connection_keywords)


class CustomDiscordBot(discord.Client):
    def __init__(self, chat_system, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = chat_system

        # Set up error handling for connection issues
        self.add_error_handler()

    def add_error_handler(self):
        # Configure logging to reduce noise from connection errors
        logger = logging.getLogger('discord')
        connection_filter = ConnectionErrorFilter()
        logger.addFilter(connection_filter)

        # Optional: Set logging level for discord-related logs
        logger.setLevel(logging.WARNING)

    async def on_disconnect(self):
        """Custom disconnect handler"""
        logging.info("Discord client disconnected. Attempting to reconnect...")

    async def on_connect(self):
        """Custom connect handler"""
        logging.info("Discord client connected successfully.")


def create_discord_bot(chat_system):
    bot = chat_system
    intents = discord.Intents.default()
    intents.message_content = True
    client = CustomDiscordBot(chat_system, intents=intents)
    debug_channel = client.get_channel(DISCORD_DEBUG_CHANNEL)

    @client.event
    async def on_ready():
        logger = logging.getLogger()
        # if global_config.DISCORD_LOGGER:
        #     logger.addHandler(DiscordLogHandler())
        logging.info('Hello {0.user} !'.format(client))
        available_personas = ', '.join(list(bot.get_persona_list().keys()))
        presence_txt = f"as {available_personas} 👀"
        await client.change_presence(
            activity=discord.Activity(name=presence_txt, type=discord.ActivityType.watching))

    @client.event
    async def on_message(message, log_chat=True):
        # ignore debug channel
        if message.channel.id == debug_channel:
            return

        logging.debug(f'{message.author}: {message.content}')

        if log_chat:
            # Log chat history to local text file
            chat_log = CHAT_LOG_LOCATION + message.guild.name + " #" + message.channel.name + ".txt"
            with open(chat_log, 'a', encoding='utf-8') as file:
                file.write(f'{message.created_at} {message.author.name}: {message.content}\n')

        # check new discord message for instance of persona name
        if message.author.id != client.user.id:
            # check for persona mention in message
            for persona_name, persona in list(bot.get_persona_list().items()):
                persona_mention = f"{persona_name}"
                logging.debug('Checking for persona name: ' + persona_name)
                if (message.content.lower().startswith(persona_mention) or
                        message.channel.name.startswith(persona_mention)):
                    if message.channel.name.startswith(persona_mention):
                        message.content = persona_mention + " " + message.content
                    logging.debug('Found persona name: ' + persona_name)
                    async with message.channel.typing():
                        # Gather context (message history) from discord
                        # Pulls a list of length GLOBAL_CONTEXT_LIMIT, is pruned later based on persona context setting #TODO: can make this more efficient by pulling the persona context limit here
                        # Formats each message to put persona name first # TODO: add persona field to preprocess_message and pass in when generating: allows messages to be used that don't start with persona name (better?)
                        # If preprocess_message with check_only=True returns True, the message is skipped as it is identified as a dev command
                        channel = client.get_channel(message.channel.id)

                        image_url = await get_image_attachments(message)

                        context = await history_gatherer(channel, message, persona_mention)

                        await set_status_streaming(persona_name)

                        # Message processing starts
                        # Check for dev commands
                        dev_response = bot.bot_logic.preprocess_message(message)
                        if dev_response is None:
                            async with channel.typing():
                                # If no dev response found, process as a bot request
                                try:
                                    response = await bot.generate_response(persona_name, message.content, context=context, image_url=image_url)
                                except TypeError as e:
                                    response = "Request failed: " + str(e)
                            await send_message(channel, response, char_limit=DISCORD_CHAR_LIMIT)
                            await reset_discord_status()

                        else:  # If dev message found, send it now and reset status
                            await send_discord_dev_message(channel, dev_response)
                            await reset_discord_status()

    async def get_image_attachments(message):
        image_url = None
        # Check for image attachments
        if any(attachment.filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')) for attachment in message.attachments):
            print("Message contains an image attachment.")
            image_url = message.attachments[0].url
        # Check for image URLs in the message content
        image_url_pattern = re.compile(r'(https?://\S+\.(?:png|jpg|jpeg|gif|bmp))', re.IGNORECASE)
        if image_url_pattern.search(message.content):
            print("Message contains an image URL.")
            image_url = image_url_pattern.search(message.content)[0]
        return image_url

    async def history_gatherer(channel, message, persona_mention): # TODO: explore using local chat logs instead of chat-platform-specific message queries (would not respect deleted messages without a bunch of work which is why I have it this way)
        context = []
        history = channel.history(before=message,
                                  limit=global_config.GLOBAL_CONTEXT_LIMIT)  # TODO: use persona-specific context limit here?
        async for msg in history:
            if msg.author is not client.user.id and msg.channel.name.startswith(persona_mention):
                msg.content = persona_mention + " " + msg.content
            # If a message begins with derpr: <persona_name> ``` the message is considered a dev message response and also skipped
            # # zero-width space is a hack used in send_dev_command to escape existing code commenting
            is_previous_dev_response = 'derpr: ' + persona_mention + ' `​``' in msg.content
            if bot.bot_logic.preprocess_message(msg, check_only=True) or is_previous_dev_response:
                continue
            else:
                context.append(
                    f"{msg.created_at.strftime('%Y-%m-%d, %H:%M:%S')}, {msg.author.name}: {msg.content}")
        return context

    async def set_status_streaming(persona_name):
        # Change discord status to 'streaming <persona>...'
        activity = discord.Activity(
            type=discord.ActivityType.streaming,
            name=persona_name + '...',
            url='https://www.twitch.tv/discordmakesmedothis')
        await client.change_presence(activity=activity)

    async def reset_discord_status():
        """ Reset discord name and status to default"""
        # await client.user.edit(username='derpr')
        # Reset discord status to 'watching'
        available_personas = ', '.join(list(chat_system.get_persona_list().keys()))
        presence_txt = f"as {available_personas} 👀"
        await client.change_presence(
            activity=discord.Activity(name=presence_txt, type=discord.ActivityType.watching))

    return client


async def send_discord_dev_message(channel, msg: str):
    """Escape discord code formatting instances, seems to require this weird hack with a zero-width space"""
    # msg.replace("```", "\```")
    formatted_msg = re.sub('```', '`\u200B``', msg)
    # Split the response into multiple messages if it exceeds 2000 characters
    chunks = split_string_by_limit(formatted_msg, DISCORD_CHAR_LIMIT-6)
    for chunk in chunks:
        try:
            await channel.send(f"```{chunk}```")
        except HTTPException as e:
            logging.error(f"An error occurred: {e}")
            pass


async def send_message(channel, msg, char_limit):
    """# Set name to currently speaking persona"""
    # await client.user.edit(username=persona_name) #  This doesn't work as name changes are rate limited to 2/hour

    # Split the response into multiple messages if it exceeds max discord message length
    chunks = split_string_by_limit(msg, char_limit)
    for chunk in chunks:
        await channel.send(f"{chunk}")
