# PawSense - Dog Breed Identification with AI

PawSense is an advanced dog breed identification and analysis system that combines image classification with Vision-Language capabilities.

## Key Features

- **Accurate Breed Identification**: Upload any dog photo to get instant breed identification with confidence scores
- **Visual Analysis**: Get detailed visual reasoning about the dog's appearance and distinctive features
- **Natural Language Q&A**: Ask questions about the detected breed or dogs in general
- **Visual Reasoning**: Understand exactly how the AI made its identification decision

## Architecture

The project consists of two main components:

- **Backend**: A FastAPI-based Python service that provides the AI capabilities:
  - MobileNetV2-based breed classification
  - BLIP Vision-Language Model for visual understanding
  - OpenVINO for optimized model inference
  - General dog knowledge Q&A

- **Frontend**: A Next.js application with a modern UI for interacting with the AI:
  - Clean, responsive interface
  - Real-time analysis
  - Multiple viewing modes for different types of analysis

## Deployment

The application is deployed on Render, with separate services for:

- **Frontend**: https://pawsense-model-avinash.onrender.com
- **Backend**: https://pawsense-api.onrender.com

## Local Development

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Docker Support

Both components include Docker support for containerized deployment:

### Backend

```bash
cd backend
docker build -t pawsense-backend .
docker run -p 8000:8000 pawsense-backend
```

### Frontend

```bash
cd frontend
docker build -t pawsense-frontend .
docker run -p 3000:3000 -e BACKEND_URL=http://localhost:8000 pawsense-frontend
```

## Technology Stack

- **Backend**: FastAPI, PyTorch, OpenVINO, BLIP, Transformers
- **Frontend**: Next.js, React, Tailwind CSS, Shadcn UI

