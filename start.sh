#!/bin/bash
# Entrypoint script for the SET Game Detector application

echo "==================== STARTING SET DETECTOR API ===================="
echo "Running on host: $(hostname)"
echo "Date/Time: $(date)"
echo "Python version: $(python --version)"
echo ""

# Set default values for environment variables if not provided
export PORT=${PORT:-5000}
export MAX_WORKERS=${MAX_WORKERS:-3}
export MAX_SESSIONS=${MAX_SESSIONS:-30}
export SESSION_TTL=${SESSION_TTL:-600}
export CLEANUP_INTERVAL=${CLEANUP_INTERVAL:-120}
export MAX_MEMORY_PERCENT=${MAX_MEMORY_PERCENT:-80}
export ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-"*"}
export MODELS_DIR=${MODELS_DIR:-/app/models}

# Set memory limits for TensorFlow to prevent OOM errors
# Use 70% of available memory at maximum for TensorFlow
export TF_MEMORY_LIMIT_MB=$((${MEMORY_LIMIT_MB:-1536} * 7 / 10))
export TF_MEMORY_ALLOCATION="0.7"

# Configure TensorFlow for better performance in container
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Allow GPU memory growth

# Print environment for debugging
echo "Configuration:"
echo "PORT=$PORT"
echo "MAX_WORKERS=$MAX_WORKERS"
echo "MAX_SESSIONS=$MAX_SESSIONS"
echo "SESSION_TTL=$SESSION_TTL"
echo "MODELS_DIR=$MODELS_DIR"
echo "ALLOWED_ORIGINS=$ALLOWED_ORIGINS"
echo "TF_MEMORY_LIMIT_MB=$TF_MEMORY_LIMIT_MB"
echo "=================================================================="

# Pre-flight checks
echo "Running pre-flight checks..."

# Check if required directories exist
if [ -d "$MODELS_DIR" ]; then
    echo "✅ Models directory found at $MODELS_DIR"
    
    # Check required subdirectories
    REQ_SUBDIRS=("Card" "Characteristics" "Shape")
    MISSING_SUBDIRS=()
    
    for subdir in "${REQ_SUBDIRS[@]}"; do
        if [ -d "$MODELS_DIR/$subdir" ]; then
            echo "  ✅ $subdir directory found"
            
            # Check for version directories
            VERSION_DIRS=$(find "$MODELS_DIR/$subdir" -maxdepth 1 -type d | wc -l)
            if [ "$VERSION_DIRS" -gt 1 ]; then
                echo "    ✅ Version directories found in $subdir"
            else
                echo "    ⚠️ No version directories found in $subdir"
                MISSING_SUBDIRS+=("$subdir version directories")
            fi
        else
            echo "  ❌ $subdir directory NOT found"
            MISSING_SUBDIRS+=("$subdir")
        fi
    done
    
    # Check specific model files
    echo "Checking for model files..."
    find "$MODELS_DIR" -type f -name "*.pt" -o -name "*.keras" -o -name "*.yaml" | sort | while read -r file; do
        file_size=$(du -h "$file" | cut -f1)
        echo "  ✅ Found $(basename "$file") ($file_size)"
    done
    
    # Print warning if subdirectories are missing
    if [ ${#MISSING_SUBDIRS[@]} -ne 0 ]; then
        echo "⚠️ WARNING: Missing directories: ${MISSING_SUBDIRS[*]}"
        echo "⚠️ Models may not load correctly!"
    fi
else
    echo "❌ ERROR: Models directory NOT found at $MODELS_DIR"
    echo "⚠️ Will attempt to find models in other locations at runtime"
fi

# Create logs directory if it doesn't exist
mkdir -p logs
chmod 777 logs

echo "Starting Gunicorn server..."
echo "=================================================================="

# Use a longer timeout for model loading
# Adjust worker timeout based on Railway's constraints
# Use a pre-loading app module to ensure workers can handle requests
exec gunicorn \
    --workers=$MAX_WORKERS \
    --threads=4 \
    --timeout=300 \
    --graceful-timeout=60 \
    --keep-alive=65 \
    --worker-class=gthread \
    --worker-tmp-dir=/tmp \
    --bind=0.0.0.0:$PORT \
    --log-level=info \
    --access-logfile=- \
    --error-logfile=- \
    --capture-output \
    --preload \
    app:app
