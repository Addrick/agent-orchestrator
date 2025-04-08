import logging
import re
import discord
from discord import HTTPException

from config import global_config
from config.global_config import *
from config.global_config import DISCORD_CHAR_LIMIT
from src.utils.message_utils import split_string_by_limit

logger = logging.getLogger()

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
            'ClientConnectorError',
            'Shard ID None has connected to Gateway'
            'Shard ID None has successfully RESUMED'
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
        discord_logger = logging.getLogger('discord')
        connection_filter = ConnectionErrorFilter()
        discord_logger.addFilter(connection_filter)

        discord_logger.setLevel(logging.WARNING)

    async def on_disconnect(self):
        """Custom disconnect handler"""
        logging.debug("Discord client disconnected. Attempting to reconnect...")

    async def on_connect(self):
        """Custom connect handler"""
        logging.info("Discord client connected successfully.")


async def get_image_attachments(message):
    """Gets image attachments or URLs from a message."""
    image_url = None
    # Check for image attachments
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                logger.info("Message contains an image attachment.")
                image_url = attachment.url
                break  # Use the first image found

    # Check for image URLs in the message content if no attachment found yet
    if image_url is None:
        image_url_pattern = re.compile(r'(https?://\S+\.(?:png|jpg|jpeg|gif|bmp))', re.IGNORECASE)
        match = image_url_pattern.search(message.content)
        if match:
            logger.info("Message contains an image URL.")
            image_url = match[0]

    return image_url


async def history_gatherer(client, channel, message, persona_mention, bot_logic, context_limit):
    """Gathers message history for context."""
    context = []
    # Use the passed context_limit
    history = channel.history(before=message, limit=context_limit)
    async for msg in history:
        # Original logic checks author ID against client ID
        is_own_message = msg.author.id == client.user.id
        author_name = "Bot" if is_own_message else msg.author.name

        processed_content = msg.content
        # Add persona prefix if necessary (consistent with original logic for non-bot messages)
        if not is_own_message and msg.channel.name.startswith(
                persona_mention) and not processed_content.lower().startswith(persona_mention.lower()):
            processed_content = persona_mention + " " + msg.content

        # Check for previous dev response markers or preprocess flags
        is_previous_dev_response = f'derpr: {persona_mention} `\u200b``' in processed_content  # zero-width space check
        if bot_logic.preprocess_message(msg, check_only=True) or is_previous_dev_response:
            continue
        else:
            context.append(
                f"{msg.created_at.strftime('%Y-%m-%d, %H:%M:%S')}, {author_name}: {processed_content}")
    # The history is gathered oldest->newest relative to the limit, reverse to get chronological order for context
    return context[::-1]


async def set_status_streaming(client, persona_name):
    """Sets the bot's status to streaming."""
    try:
        activity = discord.Activity(
            type=discord.ActivityType.streaming,
            name=persona_name + '...',
            url='https://www.twitch.tv/discordmakesmedothis')
        await client.change_presence(activity=activity)
        logger.info(f"Set status to streaming {persona_name}")
    except Exception as e:
        logger.error(f"Failed to set streaming status: {e}")


async def reset_discord_status(client, chat_system):
    """ Resets the bot's status, respecting Discord's character limit."""
    try:
        available_personas = ', '.join(list(chat_system.get_persona_list().keys()))

        # Check if the desired text exceeds the limit
        if len(available_personas) > DISCORD_STATUS_LIMIT:
            # Calculate how much to truncate (leave space for the extra added text)
            truncate_at = DISCORD_STATUS_LIMIT - 5
            presence_txt = available_personas[:truncate_at]
            logger.warning(f"Status text exceeded {DISCORD_STATUS_LIMIT} chars. Truncated.")
        else:
            presence_txt = available_personas

        # Construct the desired base status text
        formatted_presence_txt = f"as {presence_txt} 👀"

        # Set the activity using the potentially truncated text
        activity = discord.Activity(name=formatted_presence_txt, type=discord.ActivityType.watching)
        await client.change_presence(activity=activity)
        logger.debug(f"Reset status to watching: {formatted_presence_txt}")

    except discord.errors.HTTPException as e:
        logger.error(f"Failed to set status due to Discord API error: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred while resetting status: {e}", exc_info=True)  # Log traceback


def create_discord_bot(chat_system):
    """Creates and configures the Discord bot client."""
    bot = chat_system  # Alias for convenience
    intents = discord.Intents.default()
    intents.message_content = True
    client = CustomDiscordBot(chat_system, intents=intents)
    # We need the ID, get_channel is called later if needed
    debug_channel_id = DISCORD_DEBUG_CHANNEL

    @client.event
    async def on_ready():
        # Logger is already defined at module level
        # if global_config.DISCORD_LOGGER: # Assuming global_config is accessible
        #     logger.addHandler(DiscordLogHandler()) # Assuming DiscordLogHandler is defined/imported
        logging.info('Hello {0.user} !'.format(client))
        # Call module-level helper
        await reset_discord_status(client, chat_system)

    @client.event
    async def on_message(message, log_chat=True):
        # Check if message is in the debug channel using the ID
        if message.channel.id == debug_channel_id:
            return

        logging.debug(f'{message.author}: {message.content}')

        if log_chat:
            try:
                # Log chat history to local text file
                # Assuming CHAT_LOG_LOCATION is accessible
                chat_log = CHAT_LOG_LOCATION + message.guild.name + " #" + message.channel.name + ".txt"
                with open(chat_log, 'a', encoding='utf-8') as file:
                    file.write(f'{message.created_at} {message.author.name}: {message.content}\n')
            except Exception as e:
                logger.error(f"Failed to write to chat log: {e}")

        # Ignore messages from the bot itself
        if message.author.id == client.user.id:
            return

        # Process messages potentially mentioning a persona
        active_persona_name = None
        persona_mention_prefix = ""

        # Check if message starts with a persona name or is in a persona-named channel
        for persona_name in list(bot.get_persona_list().keys()):
            mention = f"{persona_name.lower()}"  # Use lower case for matching
            content_lower = message.content.lower()
            channel_name_lower = message.channel.name.lower()

            if content_lower.startswith(mention) or channel_name_lower.startswith(mention):
                active_persona_name = persona_name  # Store the correctly cased name
                persona_mention_prefix = f"{persona_name}"  # Store the correctly cased prefix for history
                logging.debug(f'Found persona trigger: {persona_name}')

                # Prepend persona name if triggered by channel name only
                if channel_name_lower.startswith(mention) and not content_lower.startswith(mention):
                    message.content = f"{persona_name} {message.content}"
                    logging.debug(f'Prepended persona name to message content.')
                break  # Process first matching persona

        # If a persona was triggered, proceed with generation logic
        if active_persona_name:
            try:
                async with message.channel.typing():
                    # Get channel object - potential API call
                    channel = client.get_channel(message.channel.id)
                    if not channel:
                        logger.error(f"Could not find channel with ID: {message.channel.id}")
                        return

                    # Call module-level helpers, passing necessary arguments
                    image_url = await get_image_attachments(message)
                    # Pass global_config.GLOBAL_CONTEXT_LIMIT explicitly
                    context = await history_gatherer(client, channel, message, persona_mention_prefix, bot.bot_logic,
                                                     global_config.GLOBAL_CONTEXT_LIMIT)

                    # Message processing starts
                    # Check for dev commands first (using original message object)
                    dev_response = bot.bot_logic.preprocess_message(message)

                    if dev_response is None:
                        # No dev command, generate normal response
                        await set_status_streaming(client, active_persona_name)
                        try:
                            # Use the (potentially modified) message.content
                            response = await bot.generate_response(active_persona_name, message.content,
                                                                   context=context, image_url=image_url)
                        except TypeError as e:
                            response = f"Request failed: {e}"
                        except Exception as e:
                            logger.exception(f"Error during bot.generate_response for {active_persona_name}")
                            response = "Sorry, I encountered an error while generating a response."

                        await send_message(channel, response, char_limit=DISCORD_CHAR_LIMIT)
                        await reset_discord_status(client, chat_system)  # Reset status after sending

                    else:
                        # Dev command response found
                        await send_discord_dev_message(channel, dev_response)
                        await reset_discord_status(client, chat_system)  # Reset status after sending

            except discord.errors.NotFound:
                logger.warning(f"Message {message.id} not found, likely deleted.")
            except discord.errors.Forbidden:
                logger.warning(f"Missing permissions for channel {message.channel.id} or action.")
            except Exception as e:
                logger.exception(f"An unexpected error occurred processing message {message.id}: {e}")
                # Attempt to reset status even on error
                try:
                    await reset_discord_status(client, chat_system)
                except Exception as reset_e:
                    logger.error(f"Failed to reset status after error: {reset_e}")

    return client


async def send_discord_dev_message(channel, msg: str):
    """Escape discord code formatting instances, seems to require this hack with a zero-width space"""
    # msg.replace("```", "\```")
    formatted_msg = re.sub('```', '`\u200B``', msg)
    # Split the response into multiple messages if it exceeds 2000 characters
    chunks = split_string_by_limit(formatted_msg, DISCORD_CHAR_LIMIT - 6)
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
