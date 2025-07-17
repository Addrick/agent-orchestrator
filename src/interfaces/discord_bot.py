import logging
import re
import discord
from discord import HTTPException

from config import global_config
from config.global_config import *
from config.global_config import DISCORD_CHAR_LIMIT
from src.utils.message_utils import split_string_by_limit

import logging

logger = logging.getLogger(__name__)


class CustomDiscordBot(discord.Client):
    def __init__(self, chat_system, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = chat_system


async def get_image_attachments(message):
    """Gets image attachments or URLs from a message."""
    image_url = None
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                logger.info("Message contains an image attachment.")
                image_url = attachment.url
                break
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
    history = channel.history(before=message, limit=context_limit)
    async for msg in history:
        is_own_message = msg.author.id == client.user.id
        author_name = "Bot" if is_own_message else msg.author.name
        processed_content = msg.content
        if not is_own_message and msg.channel.name.startswith(
                persona_mention) and not processed_content.lower().startswith(persona_mention.lower()):
            processed_content = persona_mention + " " + msg.content

        is_previous_dev_response = f'derpr: {persona_mention} `\u200b``' in processed_content

        # --- CHANGE 1: Pass persona_mention to preprocess_message ---
        if bot_logic.preprocess_message(persona_mention, msg, check_only=True) or is_previous_dev_response:
            continue
        else:
            context.append(
                f"{msg.created_at.strftime('%Y-%m-%d, %H:%M:%S')}, {author_name}: {processed_content}")
    return context[::-1]


async def set_status_streaming(client, persona_name):
    """Sets the bot's status to streaming."""
    try:
        activity = discord.Activity(
            type=discord.ActivityType.streaming,
            name=persona_name + '...',
            url='https://www.twitch.tv/discordmakesmedothis')
        await client.change_presence(activity=activity)
        logger.debug(f"Set status to streaming {persona_name}")
    except Exception as e:
        logger.error(f"Failed to set streaming status: {e}")


async def reset_discord_status(client, chat_system):
    """ Resets the bot's status, respecting Discord's character limit."""
    try:
        available_personas = ', '.join(list(chat_system.get_persona_list().keys()))
        if len(available_personas) > DISCORD_STATUS_LIMIT:
            truncate_at = DISCORD_STATUS_LIMIT - 5
            presence_txt = available_personas[:truncate_at]
            logger.warning(f"Status text exceeded {DISCORD_STATUS_LIMIT} chars. Truncated.")
        else:
            presence_txt = available_personas
        formatted_presence_txt = f"as {presence_txt} 👀"
        activity = discord.Activity(name=formatted_presence_txt, type=discord.ActivityType.watching)
        await client.change_presence(activity=activity)
        logger.debug(f"Reset status to watching: {formatted_presence_txt}")
    except discord.errors.HTTPException as e:
        logger.error(f"Failed to set status due to Discord API error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while resetting status: {e}", exc_info=True)


def create_discord_bot(chat_system):
    """Creates and configures the Discord bot client."""
    bot = chat_system
    intents = discord.Intents.default()
    intents.message_content = True
    client = CustomDiscordBot(chat_system, intents=intents)
    debug_channel_id = DISCORD_DEBUG_CHANNEL

    @client.event
    async def on_ready():
        logger.info('Hello {0.user} !'.format(client))
        await reset_discord_status(client, chat_system)

    @client.event
    async def on_message(message, log_chat=True):
        if message.channel.id == debug_channel_id:
            return
        logger.debug(f'{message.author}: {message.content}')
        if log_chat:
            try:
                chat_log = CHAT_LOG_LOCATION + message.guild.name + " #" + message.channel.name + ".txt"
                with open(chat_log, 'a', encoding='utf-8') as file:
                    file.write(f'{message.created_at} {message.author.name}: {message.content}\n')
            except Exception as e:
                logger.error(f"Failed to write to chat log: {e}")
        if message.author.id == client.user.id:
            return

        active_persona_name = None
        persona_mention_prefix = ""
        cleaned_user_input = message.content

        for persona_name in list(bot.get_persona_list().keys()):
            mention = f"{persona_name.lower()} "
            content_lower = message.content.lower()
            channel_name_lower = message.channel.name.lower()
            is_triggered_by_mention = content_lower.startswith(mention)
            is_triggered_by_channel = channel_name_lower.startswith(f"{persona_name.lower()}")

            if is_triggered_by_mention or is_triggered_by_channel:
                active_persona_name = persona_name
                persona_mention_prefix = f"{persona_name}"
                logger.debug(f'Found persona trigger: {persona_name}')
                if is_triggered_by_mention:
                    cleaned_user_input = message.content[len(mention):].lstrip()
                    logger.debug(f"Stripped persona mention. Cleaned input: '{cleaned_user_input}'")
                break

        if active_persona_name:
            try:
                async with message.channel.typing():
                    channel = client.get_channel(message.channel.id)
                    if not channel:
                        logger.error(f"Could not find channel with ID: {message.channel.id}")
                        return
                    image_url = await get_image_attachments(message)
                    context = await history_gatherer(client, channel, message, persona_mention_prefix, bot.bot_logic,
                                                     global_config.GLOBAL_CONTEXT_LIMIT)

                    # --- CHANGE 2: Pass active_persona_name to preprocess_message ---
                    pseudo_message = type('PseudoMessage', (), {'content': cleaned_user_input})()
                    dev_response = bot.bot_logic.preprocess_message(active_persona_name, pseudo_message)

                    if dev_response is None:
                        await set_status_streaming(client, active_persona_name)
                        try:
                            response = await bot.generate_response(active_persona_name, cleaned_user_input,
                                                                   context=context, image_url=image_url)
                        except TypeError as e:
                            response = f"Request failed: {e}"
                        except Exception as e:
                            logger.exception(f"Error during bot.generate_response for {active_persona_name}")
                            response = "Sorry, I encountered an error while generating a response."
                        await send_message(channel, response, char_limit=DISCORD_CHAR_LIMIT)
                        await reset_discord_status(client, chat_system)
                    else:
                        await send_discord_dev_message(channel, dev_response)
                        await reset_discord_status(client, chat_system)

            except discord.errors.NotFound:
                logger.warning(f"Message {message.id} not found, likely deleted.")
            except discord.errors.Forbidden:
                logger.warning(f"Missing permissions for channel {message.channel.id} or action.")
            except Exception as e:
                logger.exception(f"An unexpected error occurred processing message {message.id}: {e}")
                try:
                    await reset_discord_status(client, chat_system)
                except Exception as reset_e:
                    logger.error(f"Failed to reset status after error: {reset_e}")

    return client


async def send_discord_dev_message(channel, msg: str):
    """Escape discord code formatting instances, seems to require this hack with a zero-width space"""
    formatted_msg = re.sub('```', '`\u200B``', msg)
    chunks = split_string_by_limit(formatted_msg, DISCORD_CHAR_LIMIT - 6)
    for chunk in chunks:
        try:
            await channel.send(f"```{chunk}```")
        except HTTPException as e:
            logger.error(f"An error occurred: {e}")
            pass


async def send_message(channel, msg, char_limit):
    """# Set name to currently speaking persona"""
    chunks = split_string_by_limit(msg, char_limit)
    for chunk in chunks:
        await channel.send(f"{chunk}")