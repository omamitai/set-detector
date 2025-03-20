#!/bin/bash
# Entrypoint script for the SET Game Detector application

# Set default values for environment variables if not provided
export PORT=${PORT:-5000}
export MAX_WORKERS=${MAX_WORKERS:-3}
export MAX_SESSIONS=${MAX_SESSIONS:-30}
export SESSION_TTL=${SESSION_TTL:-600}
export CLEANUP_INTERVAL=${CLEANUP_INTERVAL:-120}
export MAX_MEMORY_PERCENT=${MAX_MEMORY_PERCENT:-80}
export ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-"*"}

# Print environment for debugging
echo "==================== STARTING SET DETECTOR API ===================="
echo "Running on host: $(hostname)"
echo "Date/Time: $(date)"
echo "Python version: $(python --version)"
echo ""
echo "Configuration:"
echo "PORT=$PORT"
echo "MAX_WORKERS=$MAX_WORKERS"
echo "MAX_SESSIONS=$MAX_SESSIONS"
echo "SESSION_TTL=$SESSION_TTL"
echo "MODELS_DIR=${MODELS_DIR:-/app/models}"
echo "ALLOWED_ORIGINS=$ALLOWED_ORIGINS"
echo "=================================================================="

# Check if required directories exist
if [ -d "${MODELS_DIR:-/app/models}" ]; then
    echo "Models directory found at ${MODELS_DIR:-/app/models}"
    echo "Contents:"
    find ${MODELS_DIR:-/app/models} -type f -name "*.pt" -o -name "*.keras" | sort
else
    echo "WARNING: Models directory not found at ${MODELS_DIR:-/app/models}"
fi

# Create logs directory if it doesn't exist
mkdir -p logs
chmod 777 logs

# Start the application using gunicorn with proper error logging
exec gunicorn \
    --workers=$MAX_WORKERS \
    --timeout=180 \
    --bind=0.0.0.0:$PORT \
    --log-level=info \
    --access-logfile=- \
    --error-logfile=- \
    --capture-output \
    app:app
