# --- Stage 1: Build the React UI ---
FROM node:20-slim AS ui-builder
WORKDIR /app/ui
# Copy package files first to leverage Docker cache
COPY src/interfaces/web_assets/derpr_ui/package*.json ./
RUN npm ci
# Copy source files and build to static assets
COPY src/interfaces/web_assets/derpr_ui/ ./
RUN npm run build

# --- Stage 2: Final Production Runner ---
# Use an official lightweight Python image
FROM python:3.14-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Ensures logs are flushed immediately (essential for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies:
# - git: app_manager.py git ops
# - curl: installs agy + claude
# - bubblewrap + socat: Claude Code's Linux OS sandbox (DP-222) — bwrap enforces
#   filesystem isolation, socat relays sandboxed network through the proxy.
# - libopus0 + ffmpeg: DP-238 voice — discord-ext-voice-recv decodes Opus via
#   libopus; ffmpeg is required by discord.py's voice stack. PyNaCl comes from pip.
RUN apt-get update && apt-get install -y \
    git \
    curl \
    bubblewrap \
    socat \
    libopus0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user matching the 'ubuntu' user (UID 1000) on the AWS host
RUN useradd -u 1000 -m botuser

# Install Antigravity CLI (agy) and Claude Code (claude) as the botuser, into
# ~/.local/bin. The claude native installer needs no node; auth for headless
# `claude -p` is supplied at runtime via CLAUDE_CODE_OAUTH_TOKEN (see compose),
# so no browser login or credentials volume is required (unlike agy).
USER botuser
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash
RUN curl -fsSL https://claude.ai/install.sh | bash
# Pre-create agy's + claude's state dirs with botuser ownership: deploy mounts a
# named volume here, and Docker's copy-on-first-use propagates this ownership so
# OAuth/cache state survives container recreation.
RUN mkdir -p /home/botuser/.gemini /home/botuser/.claude
USER root

# DP-222: claude lands in ~/.local/bin; put it on PATH so shutil.which("claude")
# resolves it (CLAUDE_CLI_PATH can still override). DISABLE_AUTOUPDATER pins the
# image's claude version — the container must not self-update under a volume.
ENV PATH="/home/botuser/.local/bin:${PATH}"
ENV DISABLE_AUTOUPDATER=1

# Copy requirements first to leverage Docker cache layers
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Copy the compiled static UI assets from Stage 1
COPY --from=ui-builder /app/ui/dist ./src/interfaces/web_assets/derpr_ui/dist

# Create a directory for persistent data (SQLite db)
RUN mkdir -p /app/data

# Ensure the new user owns the data directory so it can write to it
# (Even though we mount a volume over it, this is good practice in case the container is run standalone)
RUN chown -R botuser:botuser /app

# Switch to the non-root user
USER botuser

# Command to run the application
# We use python -m to ensure imports work correctly from the root
CMD ["python", "-m", "src.main"]
