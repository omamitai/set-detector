FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    findutils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for logs with proper permissions
RUN mkdir -p logs && chmod 777 logs

# Explicitly create models directory structure
RUN mkdir -p models/Card/16042024 \
    models/Characteristics/11022025 \
    models/Shape/15052024

# Copy models first to verify they exist
COPY models/Card/16042024/*.pt models/Card/16042024/
COPY models/Card/16042024/*.yaml models/Card/16042024/
COPY models/Characteristics/11022025/*.keras models/Characteristics/11022025/
COPY models/Shape/15052024/*.pt models/Shape/15052024/
COPY models/Shape/15052024/*.yaml models/Shape/15052024/

# Verify model files are present
RUN find /app/models -type f | sort

# Copy application code after models
COPY app.py set_detector.py ./
COPY .env ./

# Export MODELS_DIR environment variable (absolute path for certainty)
ENV MODELS_DIR=/app/models

# Expose port (Railway will override this with the PORT env var)
EXPOSE 5000

# Set environment variables with sensible defaults
ENV MAX_WORKERS=2 \
    MAX_SESSIONS=20 \
    SESSION_TTL=300 \
    CLEANUP_INTERVAL=60 \
    MAX_MEMORY_PERCENT=80

# Command to run the application
CMD gunicorn --workers=${MAX_WORKERS} --timeout=120 --bind 0.0.0.0:${PORT} app:app
