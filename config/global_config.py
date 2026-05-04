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
TEST_DATABASE_DIR = TEST_DIR / "test_data"

# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================
# Automatically detect if we are running in a test environment (Pytest sets this var)
IS_TESTING = "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("APP_ENV") == "testing"

# If testing, override core directories to ensure isolation within the tests/ folder
if IS_TESTING:
    DATA_DIR = TEST_DATABASE_DIR
    LOGS_DIR = TEST_DATABASE_DIR / "logs"
    # Create test directories immediately to avoid race conditions
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure essential local directories exist
if not IS_TESTING:
    for directory in [DATA_DIR, LOGS_DIR, CREDENTIALS_DIR]:
        directory.mkdir(exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# JSON Configuration Files
PERSONA_SAVE_FILE = DATA_DIR / "personas.json"
MODEL_SAVE_FILE = (DATA_DIR if IS_TESTING else CONFIG_DIR) / "models.json"
DEFAULT_PERSONA_SAVE_FILE = CONFIG_DIR / "default_personas.json"
SYSTEM_PERSONA_FILE = CONFIG_DIR / 'system_personas.json'

# Application Logging
CHAT_LOG_LOCATION = LOGS_DIR

# Database Paths
_default_db_path = DATA_DIR / ("test_user_memory.db" if IS_TESTING else "user_memory.db")
MEMORY_DATABASE_FILE = os.environ.get("MEMORY_DATABASE_FILE", str(_default_db_path))

# Legacy/Hardcoded test paths (maintained for backward compatibility in existing tests)
TEST_MEMORY_DATABASE_FILE = TEST_DATABASE_DIR / "test_user_memory.db"
TEST_PERSONA_SAVE_FILE = TEST_DATABASE_DIR / "test_personas.json"


# =============================================================================
# INTERFACE FLAGS
# =============================================================================
# Toggles for enabling/disabling specific application interfaces
DISCORD_BOT = True
GMAIL_BOT = False
WEB_INTERFACE = False
KOBOLD_PORT = 5002
# Persona served when kobold-lite connects without picking one explicitly.
# Overridable with KOBOLD_DEFAULT_PERSONA env var.
KOBOLD_DEFAULT_PERSONA = os.environ.get("KOBOLD_DEFAULT_PERSONA", "test_persona")
UPDATE_MODELS_ON_STARTUP = True

# =============================================================================
# DISCORD CONFIGURATION
# =============================================================================
DISCORD_CHAR_LIMIT = 2000
DISCORD_STATUS_LIMIT = 128

# Tool use limit to avoid infinite loops
MAX_TOOL_CALLS = 5
# Max cached API request payloads (for dump commands); FIFO eviction beyond this
MAX_CACHED_API_REQUESTS = 128
# Seconds before a pending tool confirmation expires (CONFIRM execution mode)
PENDING_CONFIRMATION_TIMEOUT = 300
# Channel ID for specific debug outputs (loaded from env for security)
DISCORD_DEBUG_CHANNEL = int(os.environ.get("DISCORD_DEBUG_CHANNEL", "0"))

# Channels where the bot passively logs content but does not reply unless prompted
AMBIENT_LOGGING_CHANNELS = ["general", "random", "development"]

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
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_ULTRAFAST_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_PERSONA = "You are a helpful LLM assistant."

# Token generation limits
DEFAULT_TOKEN_LIMIT = 4096

# Context window limits (number of messages)
DEFAULT_HISTORY_MESSAGES = 15
GLOBAL_HISTORY_MESSAGES = 30  # Hard cap for history sent to APIs

# Total per-persona context budget (prompt + reserved response). Matches
# kobold-lite's localsettings.max_context_length semantic so the value can
# round-trip to the slider. Old persona configs without the field default here.
DEFAULT_MAX_CONTEXT_TOKENS = 131072

# API Error Handling
EMPTY_RESPONSE_RETRIES = 3
EMPTY_RESPONSE_RETRY_DELAY = 2

# =============================================================================
# INTERNAL HELPER PERSONAS
# =============================================================================
# Persona name for model selection helper
MODEL_SELECTOR_PERSONA_NAME = "model_selector"
# Persona name for tool selection helper
TOOL_SELECTOR_PERSONA_NAME = "tool_selector"

# =============================================================================
# --- Zammad Bot Configuration ---
# =============================================================================
ZAMMAD_POLL_INTERVAL = 10
ZAMMAD_TRIAGE_TAG = "autotriaged"
TRIAGE_GLOBAL_HISTORY_COUNT = 5
TRIAGE_USER_HISTORY_COUNT = 3
TRIAGE_MAX_CONTEXT_CHARS = 100000  # ~25k tokens

TRIAGE_SCOUT_NAME = "triage_scout"
TRIAGE_SUMMARIZER_NAME = "triage_summarizer"
TRIAGE_ANALYST_NAME = "triage_analyst"
TRIAGE_FILTER_NAME = "triage_filter"


ZAMMAD_BOT_EMAIL = "autotriage@bot.local"
ZAMMAD_BOT_FIRSTNAME = "autotriage"
ZAMMAD_BOT_LASTNAME = "LLM"

# =============================================================================
# --- Dispatch Agent Configuration ---
# =============================================================================
DISPATCH_POLL_INTERVAL = 30
DISPATCH_TRIAGE_TAG = ZAMMAD_TRIAGE_TAG  # tickets must have this tag
DISPATCH_DISPATCHED_TAG = "ai_dispatched"  # tag applied after dispatch
DISPATCH_PERSONA_NAME = "dispatch_analyst"

# =============================================================================
# --- Reminder Agent Configuration ---
# =============================================================================
REMINDER_POLL_INTERVAL = 3600  # Default to 1 hour
REMINDER_STALE_THRESHOLD_HOURS = 24
REMINDER_SENT_TAG = "ai_reminder_sent"
# =============================================================================
# --- Long-Term Memory Configuration ---
# =============================================================================
MEMORY_RETRIEVAL_ENABLED = True
MEMORY_MAX_SUMMARIES_IN_CONTEXT = 5

LOCAL_LLM_URL = "http://omen:5001/v1"

# =============================================================================
# --- API Rate Limiting ---
# =============================================================================
# All limits are overridable via environment variables so they can be adjusted
# without a redeploy if a provider quietly changes their quotas.
# Note: Google Search Grounding has a separate quota not tracked here —
# the 429 short-circuit handles grounding overruns.
#
# Gemini 2.5 (confirmed 2026-03): 5 RPM, 20 RPD
RATE_LIMIT_GEMINI_25_RPM = int(os.environ.get("RATE_LIMIT_GEMINI_25_RPM", "5"))
RATE_LIMIT_GEMINI_25_RPD = int(os.environ.get("RATE_LIMIT_GEMINI_25_RPD", "20"))
# Gemini 3.1 (confirmed 2026-03): 15 RPM  (other gemini-3.x models have quota 0)
RATE_LIMIT_GEMINI_3_RPM  = int(os.environ.get("RATE_LIMIT_GEMINI_3_RPM",  "15"))
# Gemma 3 free tier (confirmed 2026-04): 30 RPM
RATE_LIMIT_GEMMA_3_RPM   = int(os.environ.get("RATE_LIMIT_GEMMA_3_RPM",   "30"))
# Gemma 4 free tier (confirmed 2026-04): 15 RPM
RATE_LIMIT_GEMMA_4_RPM   = int(os.environ.get("RATE_LIMIT_GEMMA_4_RPM",   "15"))
RATE_LIMIT_GEMMA_4_RPD   = int(os.environ.get("RATE_LIMIT_GEMMA_4_RPD",   "1500"))
RATE_LIMIT_GEMMA_4_TPR   = int(os.environ.get("RATE_LIMIT_GEMMA_4_RPD",   "256000"))

# OpenAI and Anthropic — set generously; adjust if you hit 429s on those providers.
RATE_LIMIT_OPENAI_RPM    = int(os.environ.get("RATE_LIMIT_OPENAI_RPM",    "60"))
RATE_LIMIT_ANTHROPIC_RPM = int(os.environ.get("RATE_LIMIT_ANTHROPIC_RPM", "50"))

# Models
EMBEDDING_MODEL = 'gemini-embedding-001'
EMBEDDING_DIMENSION = 3072

# Rate Limits (Google AI Studio tracks embeddings *per item*, not per HTTP request)
GEMINI_EMBEDDING_001_RPM = 100
GEMINI_EMBEDDING_001_TPM = 30000
GEMINI_EMBEDDING_001_RPD = 1000