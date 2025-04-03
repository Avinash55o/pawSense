FROM python:3.9-slim

# Set the project name as a label
LABEL project="PawSense"

WORKDIR /app

# Install system dependencies required for OpenVINO and PIL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files
COPY main.py .
COPY vlm_extension.py .
COPY breed_info.py .
COPY general_qa.py .

# Create directory for models
RUN mkdir -p models/breed_model

# Copy model files
COPY models/breed_model/mobilenetv2_dogbreeds.pth models/breed_model/

# Create directory for datasets if needed
RUN mkdir -p datasets
COPY datasets/labels.csv datasets/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 