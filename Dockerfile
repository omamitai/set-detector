FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Command to run the application with configurable worker count
CMD ["sh", "-c", "gunicorn --workers=${MAX_WORKERS:-2} --timeout=120 --bind 0.0.0.0:5000 app:app"]
