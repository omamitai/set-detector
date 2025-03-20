FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py set_detector.py ./
COPY models/ ./models/

# Create directory for logs
RUN mkdir -p logs && chmod 777 logs

# Expose port (Railway will override this with the PORT env var)
EXPOSE 5000

# Set environment variables with sensible defaults
ENV MAX_WORKERS=2 \
    MAX_SESSIONS=20 \
    SESSION_TTL=300 \
    CLEANUP_INTERVAL=60 \
    MAX_MEMORY_PERCENT=80

# Command to run the application
CMD ["sh", "-c", "gunicorn --workers=${MAX_WORKERS:-2} --timeout=120 --bind 0.0.0.0:${PORT:-5000} app:app"]
