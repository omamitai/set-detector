#!/bin/bash
# Entrypoint script for the SET Game Detector application

# Set default values for environment variables if not provided
PORT=${PORT:-5000}
MAX_WORKERS=${MAX_WORKERS:-3}
MAX_SESSIONS=${MAX_SESSIONS:-30}
SESSION_TTL=${SESSION_TTL:-600}
CLEANUP_INTERVAL=${CLEANUP_INTERVAL:-120}
MAX_MEMORY_PERCENT=${MAX_MEMORY_PERCENT:-80}

# Log environment variables for debugging
echo "Starting server on port $PORT with $MAX_WORKERS workers"
echo "MAX_SESSIONS=$MAX_SESSIONS, SESSION_TTL=$SESSION_TTL"
echo "MODELS_DIR=${MODELS_DIR:-/app/models}"

# Start the application using gunicorn
exec gunicorn --workers=$MAX_WORKERS --timeout=120 --bind 0.0.0.0:$PORT app:app
