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

# Copy application code and startup script
COPY app.py set_detector.py start.sh ./
COPY .env* ./

# Make the startup script executable
RUN chmod +x ./start.sh

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

# Set a default port (will be overridden by Railway)
ENV PORT=5000
EXPOSE 5000

# Use the startup script as the entry point
CMD ["./start.sh"]
