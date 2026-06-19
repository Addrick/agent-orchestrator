# src/interfaces/discord_bot.py

import logging
import re
import discord
import asyncio
import io
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, List

from config.global_config import DISCORD_CHAR_LIMIT, DISCORD_STATUS_LIMIT, DISCORD_DEBUG_CHANNEL, \
    AMBIENT_LOGGING_CHANNELS, GLOBAL_HISTORY_MESSAGES, PENDING_CONFIRMATION_TIMEOUT
from src.utils.message_utils import split_string_by_limit
from src.personas.store import save_personas_to_file
from src.chat_system import ChatSystem, ResponseType
from src.persona import Persona
from src.self_edit.dispatcher import DispatcherError

# THE FIX: Initialize the logger at the top of the module.
logger = logging.getLogger(__name__)


class ReconnectLogHandler(logging.Handler):
    """
    A custom logging handler that intercepts the malformed "reconnect" error
    from discord.py, logs it cleanly at the INFO level, and stops it from
    propagating to the root logger where it would crash the debugger.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == 'discord.client' and record.getMessage().startswith('Attempting a reconnect'):
            # Log a clean, informative message from our own application instead.
            logger.info("Discord client is attempting to reconnect.")


class CustomDiscordBot(discord.Client):
    def __init__(self, chat_system: 'ChatSystem', *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.chat_system: ChatSystem = chat_system

    async def send_dm(self, user_id: int, content: str) -> bool:
        """Sends a message to a user via DM, with automatic chunking."""
        try:
            # Ensure the bot is connected before checking mutual guilds
            await self.wait_until_ready()
            
            # Try cache first, then API
            user = self.get_user(user_id) or await self.fetch_user(user_id)
            
            if not user:
                logger.error(f"Could not find user {user_id} in cache or API.")
                return False

            chunks = split_string_by_limit(content, DISCORD_CHAR_LIMIT)
            for chunk in chunks:
                await user.send(chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to send DM to {user_id}: {e}")
            return False

    async def send_to_channel(self, channel_id: int, content: str) -> bool:
        """Sends a message to a specific channel, with automatic chunking."""
        try:
            # Ensure the bot is connected
            await self.wait_until_ready()
            
            channel = await self.fetch_channel(channel_id)
            if isinstance(channel, discord.abc.Messageable):
                chunks = split_string_by_limit(content, DISCORD_CHAR_LIMIT)
                for chunk in chunks:
                    await channel.send(chunk)
                return True
            else:
                logger.error(f"Channel {channel_id} is not messageable.")
                return False
        except Exception as e:
            logger.error(f"Failed to send message to channel {channel_id}: {e}")
            return False

    async def create_agent_thread(self, parent_channel_id: int, name: str) -> Optional[int]:
        """Create a standalone public thread under ``parent_channel_id`` (DP-230).

        Used for a dispatched agent's transcript + Q&A. Returns the thread id, or
        None if the parent isn't a text channel / creation fails."""
        try:
            await self.wait_until_ready()
            parent = await self.fetch_channel(parent_channel_id)
            if not isinstance(parent, discord.TextChannel):
                logger.error(f"fixr-agents parent {parent_channel_id} is not a text channel.")
                return None
            thread = await parent.create_thread(
                name=name, auto_archive_duration=1440,
                type=discord.ChannelType.public_thread,
            )
            return thread.id
        except Exception as e:
            logger.error(f"Failed to create agent thread under {parent_channel_id}: {e}")
            return None


async def _ack(channel: Optional[discord.abc.Messageable], text: str) -> None:
    """Best-effort post of a short status line into the agent thread. A failed
    ack must never break inbound routing."""
    if channel is None:
        return
    try:
        await channel.send(text)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to post agent-thread ack: {e}")


async def _react(message: Optional[discord.Message], emoji: str) -> None:
    """Best-effort reaction (e.g. mark a note-to-self as seen)."""
    if message is None:
        return
    try:
        await message.add_reaction(emoji)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to react to agent-thread message: {e}")


async def route_agent_thread_message(
    chat_system: 'ChatSystem',
    thread_id: str,
    content: str,
    *,
    channel: Optional[discord.abc.Messageable] = None,
    message: Optional[discord.Message] = None,
) -> bool:
    """Route a message posted in an agent thread straight to the agent (DP-230).

    Returns True when the message was handled here (i.e. the thread belongs to a
    dispatched agent) so the caller skips the persona/LLM path entirely — the
    whole point is a human↔agent round-trip with NO fixr LLM turn. A ``//``
    prefix is a note-to-self and is swallowed without forwarding.

    When ``channel``/``message`` are supplied (the live path), the human gets
    feedback in the thread: a 📝 reaction on a swallowed note, a ✅ ack when the
    answer resumes the agent, and a ⚠️ notice (with the reason) when it can't —
    e.g. the agent isn't waiting. Without them (unit tests) routing still works,
    just silently."""
    fixr = chat_system.get_service("fixr")
    if fixr is None:
        return False
    registry = getattr(fixr, "registry", None)
    dispatcher = getattr(fixr, "dispatcher", None)
    if registry is None or dispatcher is None:
        return False
    record = await registry.get_by_thread(thread_id)
    if record is None:
        return False  # a thread we don't own — let normal handling skip it
    text = content.strip()
    if text.startswith("//"):
        await _react(message, "📝")  # note-to-self: seen, not forwarded
        return True
    if not text:
        # Empty/attachment-only reply: nothing to forward, but tell the human
        # so an image-only answer to a parked agent isn't silently dropped.
        await _ack(channel, "⚠️ I can only forward text to the agent — please type your answer.")
        return True
    try:
        await dispatcher.answer_agent(record.agent_id, text)
        await _ack(channel, "✅ Answer received — resuming the agent…")
    except DispatcherError as e:
        # Expected business case (e.g. agent not waiting) — tell the human why.
        await _ack(channel, f"⚠️ Can't deliver that answer: {e}")
    except Exception as e:  # noqa: BLE001 — unexpected; log + generic notice
        logger.error(f"Failed to answer agent {record.agent_id} from thread: {e}")
        await _ack(channel, "⚠️ Something went wrong delivering that answer.")
    return True


async def get_image_url(message: discord.Message) -> Optional[str]:
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                return attachment.url
    url_match: Optional[re.Match[str]] = re.search(r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp))', message.content,
                                                   re.IGNORECASE)
    if url_match:
        return url_match.group(0)
    return None


async def set_status_streaming(client: discord.Client, persona_name: str) -> None:
    activity = discord.Activity(name=f'{persona_name}...', type=discord.ActivityType.streaming,
                                url='https://www.twitch.tv/placeholder')
    await client.change_presence(activity=activity)


async def reset_discord_status(client: discord.Client, chat_system: 'ChatSystem') -> None:
    personas: List[str] = list(chat_system.visible_personas().keys())
    status_text: str = f"as {', '.join(personas)} 👀"
    if len(status_text) > DISCORD_STATUS_LIMIT:
        status_text = status_text[:DISCORD_STATUS_LIMIT - 3] + "..."
    activity = discord.Activity(name=status_text, type=discord.ActivityType.watching)
    await client.change_presence(activity=activity)


async def _send_dev_response(channel: discord.abc.Messageable, msg: str, original_message: discord.Message) -> bool:
    """Send dev response in a thread attached to the original message. Returns True on success."""
    formatted_msg: str = re.sub('```', '`\u200B``', msg)
    lang_hint: str = "json" if "Last API Request Payload" in msg else ""
    limit: int = DISCORD_CHAR_LIMIT - (len(lang_hint) + 8)
    chunks: List[str] = split_string_by_limit(formatted_msg, limit)

    try:
        thread = await original_message.create_thread(name="DERPBOT", auto_archive_duration=60)
        for chunk in chunks:
            try:
                await thread.send(f"```{lang_hint}\n{chunk}```", silent=True)
            except discord.HTTPException as e:
                logger.error(f"An error occurred sending a dev response to thread: {e}")
                return False
        return True
    except discord.HTTPException as e:
        logger.error(f"Failed to create thread for dev response: {e}. Falling back to channel.")
        for chunk in chunks:
            try:
                await channel.send(f"```{lang_hint}\n{chunk}```")
            except discord.HTTPException as e2:
                logger.error(f"An error occurred sending a dev response: {e2}")
                return False
        return True


@asynccontextmanager
async def _safe_typing(channel: discord.abc.Messageable) -> AsyncIterator[None]:
    """Typing indicator that degrades gracefully on rate limit (429).

    The typing indicator is cosmetic — it should never crash the message
    handler.  If Discord returns 429 on the POST /typing endpoint, we
    log the event and let the caller proceed without the indicator.
    """
    ctx = channel.typing()
    entered = False
    try:
        await ctx.__aenter__()
        entered = True
    except discord.HTTPException as exc:
        if exc.status != 429:
            raise
        logger.debug("Typing indicator rate-limited, continuing without it.")
    try:
        yield
    finally:
        if entered:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass


def create_discord_bot(chat_system: 'ChatSystem') -> CustomDiscordBot:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True  # Required for on_message_delete
    intents.members = True   # Required for verifying mutual guilds for DMs
    client = CustomDiscordBot(chat_system, intents=intents)

    discord_client_logger = logging.getLogger('discord.client')
    discord_client_logger.addHandler(ReconnectLogHandler())
    discord_client_logger.propagate = False

    @client.event
    async def on_ready() -> None:
        guild_names = [g.name for g in client.guilds]
        logger.info(f'Logged in as {client.user}!')
        logger.info(f'Bot is currently in {len(client.guilds)} guilds: {", ".join(guild_names)}')
        await reset_discord_status(client, chat_system)

    @client.event
    async def on_message_delete(message: discord.Message) -> None:
        success: bool = await asyncio.to_thread(
            chat_system.memory_manager.suppress_message_by_platform_id, str(message.id)
        )
        if success:
            logger.info(f"Suppressed deleted message {message.id} from LLM context.")
        else:
            logger.debug(f"Message {message.id} was deleted, but not found in local DB to suppress.")

    @client.event
    async def on_message_edit(before: discord.Message, after: discord.Message) -> None:
        if after.author == client.user:
            return

        if before.content == after.content:
            return

        success: bool = await asyncio.to_thread(
            chat_system.memory_manager.handle_message_edit, str(after.id), after.content
        )
        if success:
            logger.info(f"Updated edited message {after.id} in local DB and archived history.")
        else:
            logger.debug(f"Message {after.id} was edited, but not found in local DB to update.")

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author == client.user or (
                isinstance(message.channel, discord.abc.GuildChannel) and message.channel.id == DISCORD_DEBUG_CHANNEL):
            return

        # DP-230: a message in a dispatched agent's thread goes straight to that
        # agent (answer_agent → claude --resume), bypassing the persona/LLM path.
        # Any other thread is skipped as before.
        if isinstance(message.channel, discord.Thread):
            await route_agent_thread_message(
                chat_system, str(message.channel.id), message.content,
                channel=message.channel, message=message,
            )
            return

        active_persona_name: Optional[str] = None
        cleaned_message: str = message.content
        for name in chat_system.personas.keys():
            if message.content.lower().startswith(f"{name.lower()} "):
                active_persona_name = name
                cleaned_message = message.content[len(name) + 1:].lstrip()
                break
        if active_persona_name is None:
            for name in chat_system.personas.keys():
                if isinstance(message.channel, discord.abc.GuildChannel) and message.channel.name.lower().startswith(name.lower()):
                    active_persona_name = name
                    break

        if active_persona_name:
            try:
                server_id: Optional[str] = str(message.guild.id) if message.guild else None

                # Handle dev commands before entering typing context since they
                # resolve instantly and typing can only be cleared by a channel message.
                command_result = await chat_system.bot_logic.preprocess_message(
                    active_persona_name, str(message.author.id), cleaned_message
                )
                if command_result:
                    mutated = command_result.get("mutated", False)
                    if mutated:
                        save_personas_to_file(chat_system.personas, chat_system.system_persona_names)
                    response_text = command_result["response"]
                    if response_text.startswith("FILE_RESPONSE::"):
                        parts = response_text.split("::", 2)
                        filename = parts[1]
                        file_content = parts[2]
                        file_buffer = io.BytesIO(file_content.encode('utf-8'))
                        discord_file = discord.File(fp=file_buffer, filename=filename)
                        await message.channel.send("Here is the context dump:", file=discord_file)
                    else:
                        success = await _send_dev_response(message.channel, response_text, message)
                        if mutated or not success:
                            await message.add_reaction('✅' if success else '❌')
                    await reset_discord_status(client, chat_system)
                    return

                async with _safe_typing(message.channel):
                    response_text, response_type, assistant_id, _ = await chat_system.generate_response(
                        persona_name=active_persona_name,
                        user_identifier=str(message.author.id),
                        channel=message.channel.name if isinstance(message.channel, discord.abc.GuildChannel) else "DM",
                        message=cleaned_message,
                        server_id=server_id,
                        image_url=await get_image_url(message),
                        history_limit=GLOBAL_HISTORY_MESSAGES,
                        user_display_name=message.author.display_name,
                        platform_message_id=str(message.id),
                        timestamp=message.created_at
                    )

                if response_text and response_text.startswith("FILE_RESPONSE::"):
                    # Handle special responses that are meant to be sent as file attachments.
                    # The format is "FILE_RESPONSE::filename.txt::file_content"
                    parts = response_text.split("::", 2)
                    filename = parts[1]
                    file_content = parts[2]

                    # Create a file-like object in memory to send to Discord
                    file_buffer = io.BytesIO(file_content.encode('utf-8'))
                    discord_file = discord.File(fp=file_buffer, filename=filename)
                    await message.channel.send("Here is the context dump:", file=discord_file)

                elif response_type == ResponseType.PENDING_CONFIRMATION:
                    confirm_msg = await message.channel.send(response_text)
                    await confirm_msg.add_reaction('✅')
                    await confirm_msg.add_reaction('❌')

                    def reaction_check(reaction: discord.Reaction, user: discord.User) -> bool:
                        return (user == message.author
                                and reaction.message.id == confirm_msg.id
                                and str(reaction.emoji) in ('✅', '❌'))

                    try:
                        reaction, _ = await client.wait_for(
                            'reaction_add', timeout=PENDING_CONFIRMATION_TIMEOUT, check=reaction_check
                        )
                        approved = str(reaction.emoji) == '✅'
                    except asyncio.TimeoutError:
                        approved = False
                        await confirm_msg.edit(content=response_text + "\n\n*(Confirmation timed out)*")

                    try:
                        await confirm_msg.clear_reactions()
                    except discord.HTTPException:
                        pass

                    async with _safe_typing(message.channel):
                        final_text, final_type, final_assistant_id, _ = await chat_system.resume_pending_confirmation(
                            str(message.author.id), active_persona_name, approved=approved
                        )
                    if final_text and final_text.strip():
                        chunks = split_string_by_limit(final_text, DISCORD_CHAR_LIMIT)
                        last_confirm_reply: Optional[discord.Message] = None
                        for chunk in chunks:
                            last_confirm_reply = await message.channel.send(chunk)
                        if last_confirm_reply and final_assistant_id is not None:
                            await asyncio.to_thread(
                                chat_system.memory_manager.update_platform_message_id,
                                final_assistant_id, str(last_confirm_reply.id)
                            )

                elif response_text and response_text.strip():
                    persona: Persona = chat_system.personas[active_persona_name]
                    final_reply_text: str = response_text
                    if persona.should_display_name_in_chat():
                        final_reply_text = f"**{active_persona_name}:** {response_text}"

                    chunks = split_string_by_limit(final_reply_text, DISCORD_CHAR_LIMIT)
                    last_reply_message: Optional[discord.Message] = None
                    for chunk in chunks:
                        last_reply_message = await message.channel.send(chunk)

                    if last_reply_message and assistant_id is not None:
                        await asyncio.to_thread(
                            chat_system.memory_manager.update_platform_message_id,
                            assistant_id, str(last_reply_message.id)
                        )
                await reset_discord_status(client, chat_system)
                return
            except Exception as e:
                logger.error(f"An unexpected error occurred in on_message: {e}", exc_info=True)
                await message.channel.send("A critical error occurred. Please check the logs.")
                await reset_discord_status(client, chat_system)
                return

        if isinstance(message.channel, discord.abc.GuildChannel) and message.channel.name.lower() in [
                c.lower() for c in AMBIENT_LOGGING_CHANNELS]:
            server_id = str(message.guild.id) if message.guild else None
            await asyncio.to_thread(
                chat_system.memory_manager.log_message,
                user_identifier=str(message.author.id),
                persona_name="ambient",
                channel=message.channel.name,
                author_role='user',
                author_name=message.author.display_name,
                content=message.content,
                timestamp=message.created_at,
                platform_message_id=str(message.id),
                server_id=server_id
            )

    return client
