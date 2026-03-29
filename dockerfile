# Use an official lightweight Python image
FROM python:3.14-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Ensures logs are flushed immediately (essential for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (Git is needed for your app_manager.py, though Docker-native updates are better)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache layers
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a directory for persistent data (SQLite db)
RUN mkdir -p /app/data

# --- SECURITY ADDITIONS ---
# Create a non-root user matching the 'ubuntu' user (UID 1000) on the AWS host
RUN useradd -u 1000 -m botuser

# Ensure the new user owns the data directory so it can write to it
# (Even though we mount a volume over it, this is good practice in case the container is run standalone)
RUN chown -R botuser:botuser /app

# Switch to the non-root user
USER botuser
# --------------------------

# Command to run the application
# We use python -m to ensure imports work correctly from the root
CMD ["python", "-m", "src.main"]
