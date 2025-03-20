FROM python:3.9-slim

WORKDIR /app

# Install system dependencies with cleanup in the same layer to reduce image size
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    findutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for logs with proper permissions
RUN mkdir -p logs && chmod 777 logs

# Create models directory structure
RUN mkdir -p models/Card/16042024 \
    models/Characteristics/11022025 \
    models/Shape/15052024

# Copy application code
COPY app.py set_detector.py ./
COPY .env* ./

# Copy models (in separate steps for better caching)
COPY models/Card/16042024/*.pt models/Card/16042024/
COPY models/Card/16042024/*.yaml models/Card/16042024/
COPY models/Characteristics/11022025/*.keras models/Characteristics/11022025/
COPY models/Shape/15052024/*.pt models/Shape/15052024/
COPY models/Shape/15052024/*.yaml models/Shape/15052024/

# Log for debugging purposes
RUN echo "Verifying model files..." && find models -type f | sort

# Set environment variables
ENV MAX_WORKERS=2 \
    MAX_SESSIONS=20 \
    SESSION_TTL=300 \
    CLEANUP_INTERVAL=60 \
    MAX_MEMORY_PERCENT=80 \
    MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Railway will set the PORT environment variable
EXPOSE ${PORT:-5000}

# Use a proper startup script to handle the PORT environment variable
CMD gunicorn --workers=${MAX_WORKERS} --timeout=120 --bind 0.0.0.0:${PORT:-5000} app:app
