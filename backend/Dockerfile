FROM python:3.9-slim

# Set the project name as a label
LABEL project="PawSense"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory (new structure)
COPY app/ ./app/

# Copy additional files if needed
COPY vlm_extension.py .

# Create directory for model cache
RUN mkdir -p /tmp/models

# Set environment variables
ENV MODEL_CACHE_DIR=/tmp/models
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with new structure
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]