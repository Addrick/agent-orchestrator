import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Resolve the project root directory relative to this config file.
# This ensures file paths remain correct regardless of the execution context (local vs Docker).
CONFIG_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CONFIG_DIR.parent

# Core directories
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CREDENTIALS_DIR = PROJECT_ROOT / "credentials"
TEST_DIR = PROJECT_ROOT / "tests"

# Ensure essential local directories exist
for directory in [DATA_DIR, LOGS_DIR, CREDENTIALS_DIR]:
    directory.mkdir(exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# JSON Configuration Files
PERSONA_SAVE_FILE = DATA_DIR / "personas.json"
TEST_PERSONA_SAVE_FILE = DATA_DIR / "test_personas.json"
MODEL_SAVE_FILE = CONFIG_DIR / "models.json"
DEFAULT_PERSONA_SAVE_FILE = CONFIG_DIR / "default_personas.json"

# Application Logging
CHAT_LOG_LOCATION = LOGS_DIR

# Database Paths
# Allows override via environment variables for Docker volume mapping
_default_db_path = DATA_DIR / "user_memory.db"
MEMORY_DATABASE_FILE = os.environ.get("MEMORY_DATABASE_FILE", str(_default_db_path))

# Test Database Paths
TEST_DATABASE_DIR = TEST_DIR / "test_data"
TEST_MEMORY_DATABASE_FILE = TEST_DATABASE_DIR / "test_user_memory.db"

# =============================================================================
# INTERFACE FLAGS
# =============================================================================
# Toggles for enabling/disabling specific application interfaces
DISCORD_BOT = True
GMAIL_BOT = True
UPDATE_MODELS_ON_STARTUP = True

# =============================================================================
# DISCORD CONFIGURATION
# =============================================================================
DISCORD_CHAR_LIMIT = 2000
DISCORD_STATUS_LIMIT = 128

GEMINI_EMPTY_RESPONSE_RETRIES = 2  # Number of times to retry on a valid but empty response
# The number of times to retry any LLM provider on a valid but empty response
EMPTY_RESPONSE_RETRIES = 2
# The short delay (in seconds) between empty response retries
EMPTY_RESPONSE_RETRY_DELAY = 0.5
# Tool use limit to avoid infinite loops
MAX_TOOL_CALLS = 5
# Channel ID for specific debug outputs (loaded from env for security)
DISCORD_DEBUG_CHANNEL = int(os.environ.get("DISCORD_DEBUG_CHANNEL", "0"))

# Channels where the bot passively logs content but does not reply unless prompted
AMBIENT_LOGGING_CHANNELS = ["general", "random", "development"]

# Channels that trigger Zammad ticket creation logic
SUPPORT_CHANNELS = ["support", "helpdesk", "it-requests"]

# =============================================================================
# GMAIL & PUBSUB CONFIGURATION
# =============================================================================
# Credentials paths (overridable for Docker secrets)
GMAIL_CREDENTIALS_FILE = os.environ.get("GMAIL_CREDENTIALS_FILE", str(CREDENTIALS_DIR / "credentials.json"))
GMAIL_TOKEN_FILE = os.environ.get("GMAIL_TOKEN_FILE", str(CREDENTIALS_DIR / "token.json"))

# Google Cloud Pub/Sub settings for Gmail watch
GMAIL_PROJECT_ID = os.environ.get("GMAIL_PROJECT_ID", "derpr-production")
GMAIL_PUBSUB_TOPIC = os.environ.get("GMAIL_PUBSUB_TOPIC", "projects/derpr-production/topics/gmail-watch")
GMAIL_PUBSUB_SUBSCRIPTION_ID = os.environ.get("GMAIL_PUBSUB_SUBSCRIPTION_ID", "gmail-watch-sub")

# Email Security Filters
BLOCK_EXTERNAL_SENDER_REPLIES = True
ALLOWED_SENDER_LIST = [
    "tech-ops.it"
]

# =============================================================================
# LLM ENGINE SETTINGS
# =============================================================================
DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_ULTRAFAST_MODEL_NAME = "gemini-2.0-flash-lite"
DEFAULT_PERSONA = "You are a helpful assistant."

# Token generation limits
DEFAULT_TOKEN_LIMIT = 4096

# Context window limits (number of messages)
DEFAULT_CONTEXT_LIMIT = 15
GLOBAL_CONTEXT_LIMIT = 30  # Hard cap for history sent to APIs

# API Error Handling
EMPTY_RESPONSE_RETRIES = 3
EMPTY_RESPONSE_RETRY_DELAY = 2

# =============================================================================
# INTERNAL HELPER PERSONAS
# =============================================================================
# Persona name for model selection helper
MODEL_SELECTOR_PERSONA_NAME = "model_selector"

# =============================================================================
# --- Zammad Bot Configuration ---
# =============================================================================
ZAMMAD_BOT_ENABLED = False
ZAMMAD_POLL_INTERVAL = 10
ZAMMAD_TRIAGE_TAG = "ai_triaged"
TRIAGE_PERSONA_NAME = "triage"

# --- Local LLM Configuration ---
# Defaulting to your LAN IP as requested for the test setup
LOCAL_LLM_URL = "http://omen:5001/v1"
