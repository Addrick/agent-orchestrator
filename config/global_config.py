import json
import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Resolve the project root directory relative to this config file.
# This ensures file paths remain correct regardless of the execution context (local vs Docker).
CONFIG_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CONFIG_DIR.parent

# Load environment variables from .env file at the project root
load_dotenv(PROJECT_ROOT / ".env")


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
WEB_INTERFACE = os.environ.get("WEB_INTERFACE", "False").lower() in ("true", "1", "yes", "on")
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

# DP-118: `ingest_path` agent tool — global kill switch + hash-cache location.
# Set INGEST_PATH_ENABLED=0 to disable the tool everywhere. Per-persona gating
# still happens via `enabled_tools`; this is the emergency off-switch.
INGEST_PATH_ENABLED: bool = os.environ.get("INGEST_PATH_ENABLED", "1").lower() in ("1", "true", "yes", "on")
INGEST_CACHE_DIR: Path = Path(os.environ.get("INGEST_CACHE_DIR", str(DATA_DIR / "ingest_cache")))
# Channel ID for specific debug outputs (loaded from env for security)
DISCORD_DEBUG_CHANNEL = int(os.environ.get("DISCORD_DEBUG_CHANNEL", "0"))

# DP-277: static operator token for the portal's control plane (persona
# PATCH/create, dev_command, confirm, interaction edits — every non-GET
# adapter route outside the data-plane allowlist). Presented as
# "Authorization: Bearer <token>" (or X-Derpr-Token). Compared with
# secrets.compare_digest. EMPTY = control plane LOCKED (fail closed): set
# this env var to enable portal-side configuration at all. Never inject
# this value into any prompt, persona, or tool result.
DERPR_CONTROL_TOKEN = os.environ.get("DERPR_CONTROL_TOKEN", "")

# DP-277: bind address for the kobold engine adapter (:5003). Defaults to
# 0.0.0.0 because in the prod deploy the app runs INSIDE a container and its
# port is reached via Docker publishing (`-p 5004:5003`) from the Caddy TLS
# front — a container-loopback bind would make the published port unreachable
# (see memory infrastructure/derpr-caddy-voice-https-5003-5004). The network
# boundary here is Docker port publishing + Caddy, not the app bind. Override
# to 127.0.0.1 only for a bare-metal/loopback-fronted deploy.
KOBOLD_ADAPTER_HOST = os.environ.get("KOBOLD_ADAPTER_HOST", "0.0.0.0")

# DP-277: Discord origins allowed to run control-plane dev commands
# (add/delete/set/trust/untrust/remember/update_models). Entry format:
# "server_id[/channel_id[/author_id]]", comma-separated; missing or "*"
# components match anything at that level. Parsed by
# src.origin.parse_operator_allowlist; matched against gateway-asserted ids
# (unforgeable), never message content. Default is the operator's home server
# (whole-server grant approved 2026-07-04, see memory task DP-277); set
# OPERATOR_ALLOWLIST="" to lock Discord control commands off entirely.
OPERATOR_ALLOWLIST = os.environ.get("OPERATOR_ALLOWLIST", "347812763093172225")

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
DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_ULTRAFAST_MODEL_NAME = "gemini-2.5-flash-lite"
# Resolved by the "default_agent_model" sentinel in persona model_name —
# the one knob that moves every agent/system persona pointed at it together.
DEFAULT_AGENT_MODEL = "agy-flash"
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


# DP-288 Phase 1: content classification + phishing quarantine
CONTENT_CLASSIFIER_NAME = "content_classifier"
CLASSIFIER_MAX_CONTENT_CHARS = 8000
SECURITY_REPORT_TAG = "security-report"      # user-reported phishing
PHISHING_SUSPECT_TAG = "phishing-suspect"    # ticket itself may be a phish
QUARANTINE_TAGS = (SECURITY_REPORT_TAG, PHISHING_SUSPECT_TAG)
PHISHING_SUSPECT_MIN_CONFIDENCE = 0.6  # below this a suspect verdict is note-only

# DP-292 Phase 2: content-date anchoring for document ingest.
# Regex extraction always runs; the LLM fallback (date_tagger persona) fires
# only when regex finds no date and DATE_TAGGER_ENABLED is on.
DATE_TAGGER_NAME = "date_tagger"
DATE_TAGGER_ENABLED: bool = os.environ.get("DATE_TAGGER_ENABLED", "1").lower() in ("1", "true", "yes", "on")
DATE_EXTRACTION_MAX_CHARS = 20000  # body head scanned for dates / sent to tagger

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
# --- Managr Agent Configuration (DP-280, Phase 0: read-only) ---
# =============================================================================
MANAGR_PLANNER_NAME = "managr_planner"
MANAGR_STALE_ANALYST_NAME = "managr_stale_analyst"
MANAGR_PATTERN_ANALYST_NAME = "managr_pattern_analyst"
MANAGR_BOARD_TICKET_LIMIT = 50
MANAGR_MAX_BOARD_CHARS = 60000   # cap on the board snapshot fed to personas
MANAGR_MAX_BRIEF_CHARS = 8000    # cap on each analyst brief fed to the planner
# Detail tier (DP-288 Phase 2): a subset of open tickets gets an expanded
# fetch (first article + last two) so the planner reasons over real content,
# not just title-lines. Prioritized stale-first (oldest last_update). Tickets
# under any QUARANTINE_TAGS tag are excluded unconditionally — bait text must
# never reach the planner prompt (the whole point of Phase 1).
MANAGR_DETAIL_TICKET_LIMIT = 20  # open tickets given the expanded content fetch
MANAGR_MAX_ARTICLE_CHARS = 600   # per-article clip within the detail tier
# Peer agents whose recent actions are summarized into the board snapshot
MANAGR_PEER_AGENTS = ("zammad_bot", "dispatch", "reminder")
MANAGR_PEER_ACTION_LIMIT = 5
# Phase 1 proposal queue (DP-282)
MANAGR_PROPOSAL_TTL_DAYS = 7          # GC for pending rows managr stops reaffirming (DP-290)
MANAGR_MAX_PROPOSALS_PER_CYCLE = 10   # cap on proposals stored per planning cycle
# Self-managing queue (DP-290): pending proposals injected into the
# proposal-extraction call for reaffirm/revise/withdraw dispositions.
# Separate budget from the reviewed-outcomes section so pending rows never
# crowd out the denials the planner learns from.
MANAGR_PENDING_PROPOSAL_LIMIT = 15
# Standing orders (DP-281)
MANAGR_STANDING_ORDERS_LIMIT = 20     # newest active orders injected per planning cycle
# =============================================================================
# --- Long-Term Memory Configuration ---
# =============================================================================
MEMORY_RETRIEVAL_ENABLED = True
MEMORY_MAX_SUMMARIES_IN_CONTEXT = 5

SEMANTIC_BACKEND = os.environ.get("SEMANTIC_BACKEND", "sqlite") # Literal["sqlite", "hindsight"]
HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL", "http://10.0.0.70:8888")
HINDSIGHT_LLM_MODEL = os.environ.get("HINDSIGHT_LLM_MODEL", "qwen2.5-32b")

LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://10.0.0.70:5001/v1")

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

# Antigravity (agy) local runtime — runs on the user's OAuth tier via the local
# binary, so the ceiling is the OAuth account's, not a free API quota. Kept
# conservative; the spawn-per-call cost is the practical limiter.
RATE_LIMIT_AGY_RPM       = int(os.environ.get("RATE_LIMIT_AGY_RPM",       "15"))

# Workspace persistence settings for the agy provider
AGY_PERSISTENT_WORKSPACES = os.environ.get("AGY_PERSISTENT_WORKSPACES", "True").lower() in ("true", "1", "yes", "on")
AGY_WORKSPACE_MODE = os.environ.get("AGY_WORKSPACE_MODE", "persona") # 'persona' or 'global'
AGY_WORKSPACES_DIR = DATA_DIR / "workspaces"

# Run agy under its built-in OS-level sandbox (--sandbox: nsjail on Linux,
# sandbox-exec on macOS). Defense-in-depth while the prompt still forbids
# agy's own tools; required before that restriction is ever lifted.
AGY_SANDBOX = os.environ.get("AGY_SANDBOX", "True").lower() in ("true", "1", "yes", "on")

# Claude Code (cc) local runtime (DP-222) — `cc-*` models route through the
# local `claude -p` headless CLI on the user's OAuth/subscription tier, running
# as an autonomous agent with its OWN sandboxed tools (vs the agy route, which
# clamps tools off and uses derpr's <tool_call> text protocol). Structural
# parity with the agy provider: subprocess-per-call, one-shot, POSIX-only,
# per-persona persistent workspace, dedicated rate limiter.
RATE_LIMIT_CC_RPM = int(os.environ.get("RATE_LIMIT_CC_RPM", "15"))

# Workspace persistence (mirrors the AGY_* knobs). CC_WORKSPACE_DIR is an
# explicit override (absolute path) — set it to the derpr checkout to talk to
# Claude Code about its own codebase from any interface; otherwise per-persona
# scratch dirs under DATA_DIR/workspaces are used.
CC_PERSISTENT_WORKSPACES = os.environ.get("CC_PERSISTENT_WORKSPACES", "True").lower() in ("true", "1", "yes", "on")
CC_WORKSPACE_MODE = os.environ.get("CC_WORKSPACE_MODE", "persona")  # 'persona' or 'global'
CC_WORKSPACES_DIR = DATA_DIR / "workspaces"
CC_WORKSPACE_DIR = os.environ.get("CC_WORKSPACE_DIR")  # explicit absolute override, or None

# Run claude with --dangerously-skip-permissions (yolo) bounded by Claude
# Code's built-in OS sandbox (Seatbelt/bubblewrap). When CC_SANDBOX is on, the
# settings block below is passed via `--settings`; root is permitted because
# skip-permissions' root check is waived inside a recognized sandbox.
CC_SANDBOX = os.environ.get("CC_SANDBOX", "True").lower() in ("true", "1", "yes", "on")
# Unprivileged containers (e.g. the derpr Docker deploy) cannot mount a fresh
# /proc for bubblewrap; this bind-mounts the container's existing /proc instead.
# Only safe when the outer container already provides isolation.
CC_SANDBOX_WEAKER_NESTED = os.environ.get("CC_SANDBOX_WEAKER_NESTED", "False").lower() in ("true", "1", "yes", "on")
# Comma-separated domains the sandboxed Bash tool may reach (no domains are
# pre-allowed by default; headless runs cannot answer a domain prompt, so
# network-needing tasks must list domains here).
CC_SANDBOX_ALLOWED_DOMAINS = [
    d.strip() for d in os.environ.get("CC_SANDBOX_ALLOWED_DOMAINS", "").split(",") if d.strip()
]
# Cap on agentic turns for one headless run (--max-turns). 0/empty = no cap.
CC_MAX_TURNS = int(os.environ.get("CC_MAX_TURNS", "0"))
# Force cc-*/fixr `claude` subprocesses onto the Claude SUBSCRIPTION instead of
# the metered API. In `-p` mode the CLI prefers ANTHROPIC_API_KEY over the
# subscription OAuth token, so an inherited key (the in-process Anthropic
# provider needs one) silently bills the API. When True (default) we strip the
# API-key vars from the CLI child env so it uses CLAUDE_CODE_OAUTH_TOKEN /
# stored `/login` creds. Set False to keep API-key billing (escape hatch).
# NB: with this on, a host WITHOUT a provisioned CLAUDE_CODE_OAUTH_TOKEN makes
# the CLI fail auth (fail-loud) rather than bill the API.
CC_USE_SUBSCRIPTION = os.environ.get("CC_USE_SUBSCRIPTION", "True").lower() in ("true", "1", "yes", "on")
# Comma-separated tool allowlist for the UNSANDBOXED path (CC_SANDBOX=False).
# `--dangerously-skip-permissions` (yolo) is passed ONLY when the OS sandbox is
# the safety boundary (CC_SANDBOX=True). Without the sandbox — e.g. a native
# Windows smoke — yolo would be bare, so the engine drops it and instead passes
# these tools via `--allowedTools` (Claude Code's OS-independent permission
# system). Empty = no tools pre-allowed (default-deny; headless can't prompt, so
# tool-needing actions are refused — safe, and enough to smoke that the CLI
# runs and returns text).
CC_ALLOWED_TOOLS = [
    t.strip() for t in os.environ.get("CC_ALLOWED_TOOLS", "").split(",") if t.strip()
]

# --- DP-227: fixr base clone + per-dispatch worktrees -------------------------
# The "fixr" supervisor dispatches one detached coding agent per bug. Each agent
# runs in an isolated `git worktree` off a PRISTINE base clone of derpr's OWN
# repo (only ever `git fetch`ed, never reset/cleaned while worktrees attached),
# so parallel dispatches can't corrupt each other. The dispatch tool
# (src/self_edit/clone_manager.py) creates the worktree and points the cc-*
# engine workspace at it; engine.py stays transport-only.
#
# CC_FIXR_CLONE_DIR  — absolute path of the persistent BASE clone. Worktrees
#                      live under <CC_FIXR_CLONE_DIR>/worktrees/<bug-id>.
#                      Default: DATA_DIR/fixr_clone.
# CC_FIXR_REPO_URL   — repo to clone. None => derive from `git remote get-url
#                      origin` of the running checkout at clone time.
# CC_FIXR_BASE_REF   — ref each per-dispatch worktree branches off (origin/master).
#
# LIVE-RUN PREREQUISITES (do NOT hardcode secrets here):
#   - add `github.com,api.github.com` to CC_SANDBOX_ALLOWED_DOMAINS so the
#     sandboxed run can fetch/push and reach the GitHub API, and
#   - provide a GitHub token via the `GH_TOKEN` env var (read straight through
#     to the inherited child `gh`/`git` environment) — set it in the host
#     `.env`/an Actions secret, never in source or chat.
CC_FIXR_CLONE_DIR = os.environ.get("CC_FIXR_CLONE_DIR") or str(DATA_DIR / "fixr_clone")
# SQLite file persisting the fixr agent registry so in-flight agents survive a
# derpr restart (DP-233). On load, RUNNING/WAITING rows are marked ORPHANED.
CC_FIXR_REGISTRY_DB = os.environ.get("CC_FIXR_REGISTRY_DB") or str(DATA_DIR / "fixr_registry.db")
CC_FIXR_REPO_URL = os.environ.get("CC_FIXR_REPO_URL")  # None => derive from origin
CC_FIXR_BASE_REF = os.environ.get("CC_FIXR_BASE_REF", "origin/master")

# Supervisor wiring (the woken fixr persona + dispatched-agent model).
# CC_FIXR_PERSONA  — persona name the event bridge wakes on agent events.
# CC_FIXR_CHANNEL  — channel the bridge files fixr's woken turns under (memory scope).
# CC_FIXR_MODEL_ARG — `claude --model` arg for DISPATCHED coding agents (not the
#                     supervisor; the supervisor's model is its persona config).
# CC_FIXR_DISCORD_CHANNEL — default recipient id for fixr's send_discord tool.
CC_FIXR_PERSONA = os.environ.get("CC_FIXR_PERSONA", "fixr")
CC_FIXR_CHANNEL = os.environ.get("CC_FIXR_CHANNEL", "fixr")
CC_FIXR_MODEL_ARG = os.environ.get("CC_FIXR_MODEL_ARG", "opus")
CC_FIXR_DISCORD_CHANNEL = os.environ.get("CC_FIXR_DISCORD_CHANNEL", "")

# Direct subagent ↔ Discord channel (DP-230). Each dispatched agent gets its own
# THREAD under one auto-silenced parent channel; a human talks straight to the
# agent in-thread (question → human → answer_agent) with NO fixr LLM turn.
# CC_FIXR_AGENTS_CHANNEL_ID — Discord channel id under which per-agent threads are
#   created. Empty/0 => the direct-channel feature is OFF (events still wake fixr).
#   The channel must be pre-created (we do not auto-create channels).
# CC_FIXR_IDLE_MINUTES — minutes an unanswered `question` waits in-thread before
#   falling back to a (rare) fixr wake to auto-answer or kill the agent.
# CC_FIXR_PROGRESS_DEBOUNCE_SECONDS — coalesce window for bursty `progress`
#   events so the thread isn't one Discord message per event (rate-limit-safe).
CC_FIXR_AGENTS_CHANNEL_ID = os.environ.get("CC_FIXR_AGENTS_CHANNEL_ID", "")
CC_FIXR_IDLE_MINUTES = float(os.environ.get("CC_FIXR_IDLE_MINUTES", "10"))
CC_FIXR_PROGRESS_DEBOUNCE_SECONDS = float(
    os.environ.get("CC_FIXR_PROGRESS_DEBOUNCE_SECONDS", "1.5")
)

# Voice command pipeline (DP-238). All default-off: with VOICE_ENABLED false the
# subsystem only exposes the text-callable timer tools — no voice channel is
# joined and the optional voice deps (discord-ext-voice-recv, Moonshine) are
# never imported.
# VOICE_ENABLED — master switch for the always-listening Discord capture path.
#   NOTE: Discord voice RECEIVE is currently impossible — Discord's mandatory DAVE
#   end-to-end encryption (negotiated by discord.py >= 2.7.0) means received Opus
#   is E2E-encrypted and no Python lib can decrypt it. Use VOICE_WEB_ENABLED.
#   See memory codebase/dp238-discord-voice-recv-dead-dave-e2ee.md.
#   Because the receive path is dead, VOICE_ENABLED alone no longer makes the bot
#   join a voice channel — the join is additionally gated behind the explicit
#   VOICE_DISCORD_EXPERIMENT flag below so a stray VOICE_ENABLED can't autojoin.
# VOICE_DISCORD_EXPERIMENT — opt-in escape hatch to actually join the Discord voice
#   channel (e.g. a Stage-channel decryption experiment). Default off. Leave unset.
# VOICE_WEB_ENABLED — browser/phone push-to-talk capture at GET /voice on the web
#   interface (:5003). Records while a button is held, POSTs the utterance to
#   /voice/utterance → STT → keyword intent → timer. Needs WEB_INTERFACE on and,
#   for the fired-timer ping, VOICE_NOTIFY_CHANNEL_ID set.
# VOICE_DISCORD_CHANNEL_ID — voice channel id the bot joins to listen.
# VOICE_NOTIFY_CHANNEL_ID — text channel id a fired timer announces in (falls
#   back to the source voice channel id when unset).
# VOICE_STT_MODEL — Moonshine tier: "base" (default) or "tiny".
# VOICE_WAKEWORD — optional gate word an utterance must contain to be routed.
# VOICE_VAD_SILENCE_MS — trailing silence (ms) that closes a spoken utterance.
VOICE_ENABLED = os.environ.get("VOICE_ENABLED", "False").lower() in ("true", "1", "yes", "on")
VOICE_DISCORD_EXPERIMENT = os.environ.get("VOICE_DISCORD_EXPERIMENT", "False").lower() in ("true", "1", "yes", "on")
VOICE_WEB_ENABLED = os.environ.get("VOICE_WEB_ENABLED", "False").lower() in ("true", "1", "yes", "on")
VOICE_DISCORD_CHANNEL_ID = os.environ.get("VOICE_DISCORD_CHANNEL_ID", "")
VOICE_NOTIFY_CHANNEL_ID = os.environ.get("VOICE_NOTIFY_CHANNEL_ID", "")
VOICE_STT_MODEL = os.environ.get("VOICE_STT_MODEL", "base")
VOICE_WAKEWORD = os.environ.get("VOICE_WAKEWORD", "")
VOICE_VAD_SILENCE_MS = int(os.environ.get("VOICE_VAD_SILENCE_MS", "700"))

# Proxmox management tools (DP-262). Bot-callable ops over SSH to the pve node:
# node/guest reboot + start/stop and swapping the active koboldcpp model on the
# GPU container's :5001. All destructive tools are is_write → parked for human
# confirmation. Default-off: with no reachable key/host the tools register but
# every call returns a clear error (they never crash startup).
#
# PVE_TOOLS_ENABLED — master switch. When false the ProxmoxIntegration still
#   registers (so the startup-wiring contract holds) but tool calls short-circuit
#   with a "disabled" error instead of attempting SSH.
# PVE_SSH_HOST / PVE_SSH_USER — the Proxmox node the tools drive (NOT a guest).
#   pct/qm/systemctl/reboot all run here; the node reaches its own guests.
# PVE_SSH_KEY — path (inside the derpr container) to the private key mounted
#   read-only from the host's ~/.ssh/pve_derpr. Register this ref in the vault so
#   the egress scrubber redacts it (DP-225). Blast radius should be bounded on
#   the pve side with a forced-command authorized_keys entry (fast-follow).
# PVE_SSH_TIMEOUT — seconds before an SSH op is abandoned (ConnectTimeout + hard
#   asyncio wait). Keeps a hung node from stalling the tool loop.
# PVE_MODEL_HOST_VMID — the container id whose systemd koboldcpp units bind :5001
#   (CT101 GPU box). set_active_model/list_models run `pct exec <vmid> -- ...`.
# PVE_MODEL_UNITS — JSON object mapping a friendly model name → its systemd unit
#   on PVE_MODEL_HOST_VMID. Exactly one unit is enabled/active at a time (all bind
#   :5001). Swapping = disable --now the current, enable --now the target.
#   Defaults are the real CT101 units verified 2026-07-01 (`systemctl list-unit-files`
#   + each unit's --model path); override in .env when units change.
PVE_TOOLS_ENABLED = os.environ.get("PVE_TOOLS_ENABLED", "False").lower() in ("true", "1", "yes", "on")
PVE_SSH_HOST = os.environ.get("PVE_SSH_HOST", "10.0.0.71")
PVE_SSH_USER = os.environ.get("PVE_SSH_USER", "root")
PVE_SSH_KEY = os.environ.get("PVE_SSH_KEY", "/run/secrets/pve_derpr")
PVE_SSH_TIMEOUT = float(os.environ.get("PVE_SSH_TIMEOUT", "20"))
PVE_MODEL_HOST_VMID = os.environ.get("PVE_MODEL_HOST_VMID", "101")
PVE_MODEL_UNITS: Dict[str, str] = json.loads(
    os.environ.get("PVE_MODEL_UNITS", "")
    or json.dumps({
        "fable": "koboldcpp-fable-q6xl.service",       # Gemma-4-31B-Fable-5 Q6_K_XL (active)
        "fable-q5": "koboldcpp-fable-q5.service",       # Fable-5 Q5_K_M
        "fable-q8": "koboldcpp-fable-q8.service",       # Fable-5 Q8_0
        "fable-q4": "koboldcpp.service",                # Fable-5 Q4_K_M (generic unit name)
        "gemma": "koboldcpp-gemma-abliterated.service",  # gemma-4-31b-abliterated Q4_K_M
        "qwen-27b": "koboldcpp-qwen.service",           # Qwen3.5-27B-Uncensored Q4_K_M
        "qwen-a3b": "koboldcpp-qwen36a3b.service",      # Qwen3.6-35B-A3B-Uncensored Q4_K_M
    })
)

# =============================================================================
# --- MCP Client (DP-268) ---
# =============================================================================
# Consume external MCP tool servers (streamable-HTTP transport). Discovered
# tools register into the normal tool system under mcp__<server>__<tool> with
# restrictive default security metadata; personas must explicitly list them
# (never included by the ['*'] wildcard) and bind mcp:<server>.
#
# MCP_ENABLED — master switch. When false the mcp management tools
#   (add/remove/list_mcp_server[s]) still register so the startup-wiring
#   contract holds, but every call short-circuits with a "disabled" error and
#   no configured server is connected at startup.
# MCP_SERVERS_FILE — persisted server config (written by add_mcp_server,
#   hand-editable for per-tool security overrides). Lives in data/ (gitignored)
#   like personas.json.
# MCP_CONNECT_TIMEOUT / MCP_CALL_TIMEOUT — seconds before a server connect /
#   a single tool call is abandoned (keeps a hung server out of the tool loop).
# MCP_RECONNECT_INTERVAL — seconds between hot-reload maintenance passes
#   (reconnect dead servers; re-discover toolsets after tools/list_changed —
#   the notification wakes a pass immediately, the interval is the fallback).
#   <= 0 disables the loop: dead servers then degrade to per-call errors
#   until restart.
MCP_ENABLED = os.environ.get("MCP_ENABLED", "False").lower() in ("true", "1", "yes", "on")
MCP_SERVERS_FILE = Path(os.environ.get("MCP_SERVERS_FILE", str(DATA_DIR / "mcp_servers.json")))
MCP_CONNECT_TIMEOUT = float(os.environ.get("MCP_CONNECT_TIMEOUT", "30"))
MCP_CALL_TIMEOUT = float(os.environ.get("MCP_CALL_TIMEOUT", "120"))
MCP_RECONNECT_INTERVAL = float(os.environ.get("MCP_RECONNECT_INTERVAL", "60"))

# Models
EMBEDDING_MODEL = 'gemini-embedding-001'
EMBEDDING_DIMENSION = 3072

# Rate Limits (Google AI Studio tracks embeddings *per item*, not per HTTP request)
GEMINI_EMBEDDING_001_RPM = 100
GEMINI_EMBEDDING_001_TPM = 30000
GEMINI_EMBEDDING_001_RPD = 1000