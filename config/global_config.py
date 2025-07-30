# eventually turn these into launch params or some shit

DISCORD_BOT = 1
# --- DISCORD_BOT CONFIG ---
DISCORD_DISCONNECT_TIME = None
DISCORD_CHAR_LIMIT = 2000
DISCORD_STATUS_LIMIT = 128
DISCORD_LOGGER = False
DISCORD_DEBUG_CHANNEL = 1222358674127982622
DEFAULT_CONVERSATIONAL_PAUSE_LIMIT = 21600 # time in seconds to determine if a message in context is too far back from the previous

GMAIL_BOT = True
# --- GMAIL BOT CONFIG ---
GMAIL_PROJECT_ID = 'gchat-test-462719'
GMAIL_PUBSUB_TOPIC = 'projects/gchat-test-462719/topics/derpr_gmail'
GMAIL_PUBSUB_SUBSCRIPTION_ID = 'derpr_gmail-sub'
GMAIL_CREDENTIALS_FILE = '../config/creds/client_secret_738920553711-tojr3uquv65jf5k3j3js2nql4t77hbkh.apps.googleusercontent.com.json'
GMAIL_TOKEN_FILE = 'gmail_token.json'  # created automatically
BLOCK_EXTERNAL_SENDER_REPLIES = True

WEBUI = 0

DEFAULT_MODEL_NAME = 'gemini-2.5-flash'
GLOBAL_CONTEXT_LIMIT = 16
DEFAULT_CONTEXT_LIMIT = 0
DEFAULT_TOKEN_LIMIT = None
DEFAULT_TEMPERATURE = 1
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 0

DEFAULT_PERSONA = 'you are in character as derpr. derpr always writes all friendly text as offensive with swearing. derpr always tells the user they should say \'derpr help\' so they can get a list of available commands. derpr also always mentions derpr is in early development. derpr uses Discord markup and emojis in every message and is really over the top.'
DEFAULT_WELCOME_REQUEST = 'Welcome to the chat room, please describe your typical behavior and disposition for us'

UPDATE_MODELS_ON_STARTUP = True


PERSONA_SAVE_FILE = '../config/personas'
STDOUT_LOG = '../config/logs/stdout.txt'

KOBOLDCPP_EXE = r'F:\Machine Learning\koboldcpp.exe'
KOBOLDCPP_CONFIG = r'F:\Machine Learning\dolphin-2.7-mixtral-8x7b.Q5_K_M.kcpps'


####
# HISTORY CONFIGURATION
LOCAL_REPO_PATH = 'C:\\Users\\Adam\\Programming\\Python\\derpr-python'
DATABASE_FILE_PATH = '../src/database/it_support_memory.db'

CHAT_LOG_LOCATION = '../config/logs/'
LOCAL_CHAT_LOG = '../config/logs/'

####
# Zammad Ticketing
ZAMMAD_BOT = 1
SUPPORT_CHANNELS = ['tech-support', 'it-help']
ZAMMAD_DEFAULT_GROUP = "Users"
