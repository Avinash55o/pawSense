# PawSense Backend

This is the backend API for PawSense, a dog breed identification and analysis system that combines image classification with Vision-Language capabilities.

## Features

- Dog breed classification based on uploaded images
- Detailed breed information including characteristics, size, energy level, etc.
- Vision-Language capabilities for analyzing visual aspects of dog images
- Visual reasoning about breed identification decisions
- General dog Q&A capabilities for breed-independent questions

## Requirements

- Python 3.9+
- Dependencies listed in requirements.txt
- Docker (optional, for containerized deployment)

## Setup and Running

### Option 1: Local Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   uvicorn main:app --reload
   ```

4. Access the API at http://localhost:8000

### Option 2: Docker

1. Build and run using Docker:
   ```
   # Build the Docker image
   docker build -t pawsense-backend .
   
   # Run the container
   docker run -p 8000:8000 --name pawsense-backend-container pawsense-backend
   ```

2. Access the API at http://localhost:8000

## API Endpoints

- `POST /api/classification/predict` - Basic breed classification
- `POST /api/vision-language/analyze` - VLM analysis with descriptions
- `POST /api/vision-language/query` - Answer questions about dog in image
- `POST /api/vision-language/reasoning` - Get visual reasoning about breed identification
- `POST /api/general-qa/query` - Answer general dog questions

## Model Files

This application requires the following model:
- `models/breed_model/mobilenetv2_dogbreeds.pth` - The PyTorch model file for breed classification

The BLIP Vision-Language model will be downloaded automatically on first use.

## Development

For development, additional scripts are provided:
- `train_model.py` - For retraining the classification model
- `convert_to_ir.py` - For converting models to OpenVINO IR format
- `optimize_model.py` - For optimizing models with OpenVINO 