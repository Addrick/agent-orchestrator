# src/interfaces/discord_bot.py

import logging
import re
import discord
import asyncio
import typing
from datetime import timedelta
from typing import Optional

from config.global_config import DISCORD_CHAR_LIMIT, DISCORD_STATUS_LIMIT, CHAT_LOG_LOCATION, DISCORD_DEBUG_CHANNEL
from src.utils.message_utils import split_string_by_limit, cleanse_message_for_history
from src.chat_system import ChatSystem, ResponseType

logger = logging.getLogger(__name__)


class CustomDiscordBot(discord.Client):
    def __init__(self, chat_system: 'ChatSystem', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_system = chat_system


async def get_image_url(message: discord.Message) -> Optional[str]:
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                return attachment.url
    url_match = re.search(r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp))', message.content, re.IGNORECASE)
    if url_match:
        return url_match.group(0)
    return None


async def set_status_streaming(client: discord.Client, persona_name: str) -> None:
    activity = discord.Activity(name=f'{persona_name}...', type=discord.ActivityType.streaming,
                                url='https://www.twitch.tv/placeholder')
    await client.change_presence(activity=activity)


async def reset_discord_status(client: discord.Client, chat_system: 'ChatSystem') -> None:
    personas = list(chat_system.personas.keys())
    status_text = f"as {', '.join(personas)} 👀"
    if len(status_text) > DISCORD_STATUS_LIMIT:
        status_text = status_text[:DISCORD_STATUS_LIMIT - 3] + "..."
    activity = discord.Activity(name=status_text, type=discord.ActivityType.watching)
    await client.change_presence(activity=activity)


async def _send_dev_response(channel: discord.TextChannel, msg: str):
    formatted_msg = re.sub('```', '`\u200B``', msg)
    lang_hint = "json" if "Last API Request Payload" in msg else ""
    limit = DISCORD_CHAR_LIMIT - (len(lang_hint) + 8)
    chunks = split_string_by_limit(formatted_msg, limit)
    for chunk in chunks:
        try:
            await channel.send(f"```{lang_hint}\n{chunk}```")
        except discord.HTTPException as e:
            logger.error(f"An error occurred sending a dev response: {e}")
            pass


def create_discord_bot(chat_system: 'ChatSystem') -> CustomDiscordBot:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True  # Required for on_message_delete
    client = CustomDiscordBot(chat_system, intents=intents)

    @client.event
    async def on_ready():
        logger.info(f'Logged in as {client.user}!')
        await reset_discord_status(client, chat_system)

    @client.event
    async def on_message_delete(message: discord.Message):
        if message.author == client.user:
            return

        success = await asyncio.to_thread(
            chat_system.memory_manager.suppress_message_by_platform_id, str(message.id)
        )
        if success:
            logger.info(f"Suppressed deleted message {message.id} from LLM context.")
        else:
            logger.debug(f"Message {message.id} was deleted, but not found in local DB to suppress.")

    @client.event
    async def on_message(message: discord.Message):
        if message.author == client.user or message.channel.id == DISCORD_DEBUG_CHANNEL:
            return

        active_persona_name: Optional[str] = None
        cleaned_message: str = message.content

        for name in chat_system.personas.keys():
            if message.content.lower().startswith(f"{name.lower()} "):
                active_persona_name = name
                cleaned_message = message.content[len(name) + 1:].lstrip()
                break
            elif message.channel.name.lower().startswith(name.lower()):
                active_persona_name = name
                break

        if not active_persona_name:
            return

        try:
            async with message.channel.typing():
                user_identifier = str(message.author.id)
                channel_name = message.channel.name
                image_url = await get_image_url(message)
                user_display_name = message.author.display_name

                response_text, response_type, ticket_id = await chat_system.generate_response(
                    persona_name=active_persona_name,
                    user_identifier=user_identifier,
                    channel=channel_name,
                    message=cleaned_message,
                    image_url=image_url,
                    history_limit=20,
                    user_display_name=user_display_name
                )

                # Log the user's message as soon as the interaction starts.
                if response_type != ResponseType.DEV_COMMAND:
                    await asyncio.to_thread(
                        chat_system.memory_manager.log_message,
                        user_identifier=user_identifier,
                        persona_name=active_persona_name,
                        channel=channel_name,
                        author_role='user',
                        author_name=user_display_name,
                        content=cleaned_message,
                        timestamp=message.created_at,
                        platform_message_id=str(message.id),
                        zammad_ticket_id=ticket_id
                    )

                if response_type == ResponseType.DEV_COMMAND:
                    await _send_dev_response(message.channel, response_text)

                # The TextEngine now guarantees a non-empty response or raises an exception,
                # which is handled by the ChatSystem. This logic is now simpler.
                elif response_type == ResponseType.LLM_GENERATION:
                    persona = chat_system.personas[active_persona_name]
                    final_reply_text = response_text
                    if persona.should_display_name_in_chat():
                        final_reply_text = f"**{active_persona_name}:** {response_text}"

                    chunks = split_string_by_limit(final_reply_text, DISCORD_CHAR_LIMIT)
                    last_reply_message = None
                    for chunk in chunks:
                        last_reply_message = await message.channel.send(chunk)

                    if last_reply_message:
                        cleansed_reply = cleanse_message_for_history(response_text)
                        bot_timestamp = last_reply_message.created_at
                        if bot_timestamp <= message.created_at:
                            bot_timestamp = message.created_at + timedelta(microseconds=1)

                        await asyncio.to_thread(
                            chat_system.memory_manager.log_message,
                            user_identifier=user_identifier,
                            persona_name=active_persona_name,
                            channel=channel_name,
                            author_role='assistant',
                            author_name=active_persona_name,
                            content=cleansed_reply,
                            timestamp=bot_timestamp,
                            platform_message_id=str(last_reply_message.id),
                            zammad_ticket_id=ticket_id
                        )

            await reset_discord_status(client, chat_system)

        except Exception as e:
            logger.error(f"An unexpected error occurred in on_message: {e}", exc_info=True)
            await message.channel.send("A critical error occurred. Please check the logs.")
            await reset_discord_status(client, chat_system)

    return client
