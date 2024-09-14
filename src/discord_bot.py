import logging
import re

import discord
from discord import HTTPException

from config import global_config
from config.global_config import *
from config.global_config import DISCORD_CHAR_LIMIT
from src.utils.messages import split_string_by_limit


# Summary:
# This code implements a Discord bot using the discord.py library. The bot manages multiple personas,
# responds to messages, and handles various commands. It includes features like logging, context
# gathering, and dynamic status updates. The bot uses an external ChatSystem for generating responses.
#
# Declare all discort intents and instantiating Discord client - declaring 'all' intents for simplicity while testing
# intents = discord.Intents.all()
# client = discord.Client(intents=intents)
# guild = discord.Guild

# Import ChatSystem for use of core bot logic
# bot = ChatSystem()
# bot.load_personas_from_file(PERSONA_SAVE_FILE)

# Discord channel to dump log messages for remote debugging
# debug_channel = client.get_channel(1222358674127982622)


class CustomDiscordBot(discord.Client):
    def __init__(self, chat_system, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = chat_system


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
            for persona_name, persona in bot.get_persona_list().items():
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
                        context = []
                        history = channel.history(before=message, limit=global_config.GLOBAL_CONTEXT_LIMIT)
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
                        # Change discord status to 'streaming <persona>...'
                        activity = discord.Activity(
                            type=discord.ActivityType.streaming,
                            name=persona_name + '...',
                            url='https://www.twitch.tv/discordmakesmedothis')
                        await client.change_presence(activity=activity)

                    # Message processing starts
                    # Check for dev commands
                    dev_response = bot.bot_logic.preprocess_message(message)
                    if dev_response is None:
                        async with channel.typing():
                            # If no dev response found, process as a bot request
                            response = await bot.generate_response(persona_name, message.content, context=context)
                            await send_message(channel, response, char_limit=DISCORD_CHAR_LIMIT)
                            await reset_discord_status()

                    else:  # If dev message found, send it now and reset status
                        await send_discord_dev_message(channel, dev_response)
                        await reset_discord_status()

    async def reset_discord_status():
        """ Reset discord name and status to default"""
        # await client.user.edit(username='derpr')
        # Reset discord status to 'watching'
        available_personas = ', '.join(list(chat_system.get_persona_list().keys()))
        presence_txt = f"as {available_personas} 👀"
        await client.change_presence(
            activity=discord.Activity(name=presence_txt, type=discord.ActivityType.watching))

    return client


# # Module to forward console messages to discord
    #     # Redirect console output to discord for remote monitoring
    #     discord_console = discord_bot.DiscordConsoleOutput()
    #     # sys.stdout = discord_console
    #     # sys.stderr = discord_console
    #     # sys.excepthook = discord_console.discord_excepthook
# class DiscordConsoleOutput:
#     def __init__(self):
#         self.DISCORD_DISCONNECT_TIME = None
#
#     def write(self, msg):
#         asyncio.ensure_future(send_dev_message(debug_channel, msg))
#
#     def flush(self):
#         pass
#
#     def discord_excepthook(self, type, value, traceback):
#         if issubclass(type, ConnectionError):
#             asyncio.create_task(self.on_disconnect())
#         else:
#             error_report = f'Error logged: \n {type} \n {value} \n {traceback}'
#             asyncio.create_task(send_dev_message(debug_channel, error_report))
#
#     def on_disconnect(self):  # Disconnects must be handled as a special case so it does not flood the channel on reconnect and cause another disconnect
#         if self.DISCORD_DISCONNECT_TIME is None:
#             self.DISCORD_DISCONNECT_TIME = datetime.datetime.now()
#         else:
#             pass
#
#
# class DiscordLogHandler(logging.Handler):
#     def __init__(self):
#         super().__init__()
#         self.debug_channel = client.get_channel(1222358674127982622)
#
#     def emit(self, record):
#         log_message = self.format(record)
#         if 'ClientConnectorError' in log_message or 'We are being rate limited.' in log_message:
#             return  # Do not send message if log_message contains discord connection/rate limit errors
#         asyncio.create_task(send_dev_message(self.debug_channel, log_message))


async def send_discord_dev_message(channel, msg: str):
    """Escape discord code formatting instances, seems to require this weird hack with a zero-width space"""
    # msg.replace("```", "\```")
    formatted_msg = re.sub('```', '`\u200B``', msg)
    # Split the response into multiple messages if it exceeds 2000 characters
    chunks = split_string_by_limit(formatted_msg, DISCORD_CHAR_LIMIT)
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
