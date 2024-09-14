#     Used to emulate discord, utilities below may not work
#     StrippedMessage.__init__: Initializes a new StrippedMessage instance with the provided message content, timestamp, channel, guild, and author information.
#     Guild.__init__: Initializes a new Guild instance with an optional name, defaulting to 'local_guild'.
#     User.__init__: Initializes a new User instance with an optional name 'admin' and an id of 1 by default.
#     Channel.__init__: Initializes a new Channel instance with an optional name, defaulting to 'local_channel'.
import logging

from config.global_config import GLOBAL_CONTEXT_LIMIT, CHAT_LOG_LOCATION


class StrippedMessage:
    def __init__(self, content, timestamp, channel, guild, author):
        self.content = content
        self.created_at = timestamp
        self.channel = channel
        self.guild = guild
        self.author = author


class Guild:
    def __init__(self, name='local_guild'):
        self.name = name


class User:
    def __init__(self, name='local'):
        self.name = name
        self.id = name


class Channel:
    def __init__(self, name='local_channel'):
        self.name = name


class Client:
    def __init__(self, name='local_client'):
        self.id = name
        self.user = User(name=name)


def local_history_reader(context_limit: int):
    with open('../stuff/logs/local_guild #local_channel.txt', 'r') as file:
        lines = file.readlines()
        # Grabs last history_length number of messages from local chat history file and joins them
        context = '/n'.join(lines[-1 * (context_limit + 1):-1])
        return context


def local_history_logger(persona_name, response):
    import datetime
    with open('../stuff/logs/local_guild #local_channel.txt', 'a', encoding='utf-8') as file:
        current_time = datetime.datetime.now().time()
        response = '\n' + persona_name + ': ' + str(current_time) + ' ' + response

        file.write(response)


async def on_message(bot: ChatSystem, message, log_chat=True):
    logging.debug(f'{message.author}: {message.content}')

    if log_chat:
        # Log chat history to local text file
        chat_log = CHAT_LOG_LOCATION + 'local_terminal' + " #.txt"
        with open(chat_log, 'a', encoding='utf-8') as file:
            file.write(f'{message.created_at} {message.author.name}: {message.content}\n')

    for persona_name, persona in bot.get_persona_list().items():
        persona_mention = f"{persona_name}"
        logging.debug('Checking for persona name: ' + persona_name)
        if (message.content.lower().startswith(persona_mention) or
                message.channel.name.startswith(persona_mention)):
            if message.channel.name.startswith(persona_mention):
                message.content = persona_mention + " " + message.content
            logging.debug('Found persona name: ' + persona_name)
            async with message.channel.typing():
                # Gather context (message history) from local terminal
                context = []
                history = local_history_reader(limit=GLOBAL_CONTEXT_LIMIT)  # todo: pull context limit from persona
                for msg in history:
                    if msg.channel.name.startswith(persona_mention):
                        msg.content = persona_mention + " " + msg.content
                    # If a message begins with derpr: <persona_name> ``` the message is considered a dev message response and also skipped
                    # # zero-width space is a hack used in send_dev_command to escape existing code commenting
                    is_previous_dev_response = 'derpr: ' + persona_mention + ' `​``' in msg.content
                    if bot.bot_logic.preprocess_message(msg, check_only=True) or is_previous_dev_response:
                        continue
                    else:
                        context.append(
                            f"{msg.created_at.strftime('%Y-%m-%d, %H:%M:%S')}, {msg.author.name}: {msg.content}")

                # Message processing starts
                # Check for dev commands
                dev_response = bot.bot_logic.preprocess_message(message)
                if dev_response is None:
                    # If no dev response found, process as a bot request
                    response = await bot.generate_response(persona_name, message.content, context=context)
                    print(response)
                else:  # If dev message found, send it now and reset status
                    print(dev_response)
