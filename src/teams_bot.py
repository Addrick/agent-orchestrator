import asyncio
import logging
import re

from botbuilder.core import BotFrameworkAdapter, TurnContext
from botbuilder.schema import Activity, ActivityTypes
from botframework.connector.auth import MicrosoftAppCredentials

from src import utils
from config import global_config
from src.app_manager import restart_app, stop_app
from src.chat_system import ChatSystem
from config.global_config import *

# This rewritten version attempts to provide equivalent functionality using the Microsoft Bot Framework, which is typically used for creating bots for Microsoft Teams. Here are some key points to note:
#
# The Discord-specific client is replaced with a BotFrameworkAdapter.
# Event handlers like on_ready are not directly applicable in Teams. Instead, we use a main on_turn function to handle incoming activities.
# Discord's concept of "presence" (status) doesn't have a direct equivalent in Teams, so those parts have been removed.
# Message history retrieval is significantly different in Teams. I've included a placeholder function get_conversation_history, but you'll need to implement a custom solution for this.
# Channel and user IDs work differently in Teams. You'll need to adjust how you identify and interact with specific channels or users.
# The character limit for messages in Teams (TEAMS_CHAR_LIMIT) is not a character limit but is a size limit of 40kB
# Error handling and logging have been adapted to work with the Teams structure, but the core functionality remains similar.
#
# Remember to replace placeholder values like YOUR_DEBUG_CHANNEL_ID, APP_ID, and APP_PASSWORD with your actual Microsoft Teams bot credentials and channel IDs.

# Initialize the bot
credentials = MicrosoftAppCredentials(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(credentials)

bot = ChatSystem()
# bot.load_personas_from_file(PERSONA_SAVE_FILE)

debug_channel_id = "YOUR_DEBUG_CHANNEL_ID"

async def on_turn(turn_context: TurnContext):
    if turn_context.activity.type == ActivityTypes.message:
        await on_message(turn_context)

async def on_message(turn_context: TurnContext, log_chat=True):
    message = turn_context.activity

    # Ignore messages from the debug channel
    if message.channel_id == debug_channel_id:
        return

    logging.debug(f'{message.from_property.name}: {message.text}')

    if log_chat:
        # Log chat history
        chat_log = CHAT_LOG_LOCATION + f"{message.channel_data['team']['name']} #{message.channel_data['channel']['name']}.txt"
        with open(chat_log, 'a', encoding='utf-8') as file:
            file.write(f'{message.timestamp} {message.from_property.name}: {message.text}\n')

    if message.from_property.id != credentials.microsoft_app_id:
        # check message for instance of persona name
        for persona_name, persona in bot.get_persona_list().items():
            persona_mention = f"{persona_name}"
            logging.debug('Checking for persona name: ' + persona_name)
            if (message.text.lower().startswith(persona_mention) or
                    message.channel_data['channel']['name'].startswith(persona_mention)):
                if message.channel_data['channel']['name'].startswith(persona_mention):
                    message.text = persona_mention + " " + message.text
                logging.debug('Found persona name: ' + persona_name)

                # Gather context (message history) from Teams
                # Note: Teams API doesn't provide a direct way to fetch message history
                # You might need to implement a custom solution to store and retrieve context
                context = await get_conversation_history(turn_context, global_config.GLOBAL_CONTEXT_LIMIT)

                # Message processing starts
                # Check for dev commands
                dev_response = bot.bot_logic.preprocess_message(message)
                if dev_response is None:
                    response = await bot.generate_response(persona_name, message.text, turn_context, bot, adapter, context)
                else:
                    await send_dev_message(turn_context, dev_response)
                    if dev_response == f'App restarting...':
                        restart_app()
                    if dev_response == f'App stopping...':
                        stop_app()

async def get_conversation_history(turn_context: TurnContext, limit: int):
    # This is a placeholder function. Teams API doesn't provide a direct way to fetch message history.
    # You'll need to implement a custom solution to store and retrieve conversation history.
    # This might involve using a database or other storage mechanism.
    return []

class TeamsConsoleOutput:
    def __init__(self):
        pass

    def write(self, msg):
        asyncio.ensure_future(send_dev_message(TurnContext(adapter, Activity(channel_id=debug_channel_id)), msg))

    def flush(self):
        pass

    def teams_excepthook(self, type, value, traceback):
        error_report = f'Error logged: \n {type} \n {value} \n {traceback}'
        asyncio.create_task(send_dev_message(TurnContext(adapter, Activity(channel_id=debug_channel_id)), error_report))

class TeamsLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        log_message = self.format(record)
        asyncio.create_task(send_dev_message(TurnContext(adapter, Activity(channel_id=debug_channel_id)), log_message))

async def send_dev_message(turn_context: TurnContext, msg: str):
    formatted_msg = re.sub('```','`\u200B``', msg)
    chunks = utils.split_string_by_limit(formatted_msg, TEAMS_CHAR_LIMIT)
    for chunk in chunks:
        await turn_context.send_activity(f"```{chunk}```")

async def send_message(turn_context: TurnContext, msg):
    chunks = utils.split_string_by_limit(msg, TEAMS_CHAR_LIMIT)
    for chunk in chunks:
        await turn_context.send_activity(f"{chunk}")

# Main entry point for the bot
app = adapter.server_processes

if __name__ == "__main__":
    try:
        app.run(host='localhost', port=3978)
    except Exception as e:
        raise e