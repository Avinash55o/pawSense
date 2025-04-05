from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from openvino.runtime import Core
from pathlib import Path
import logging
import traceback
from breed_info import get_breed_info, get_all_breeds
from typing import Optional, List, Dict, Any, Union
import time
import re
from torchvision import transforms
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our VLM extension
from vlm_extension import DogBreedVLM
# Import our general QA model
from general_qa import get_qa_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with metadata
app = FastAPI(
    title="PawSense API",
    description="Dog breed identification and analysis with Vision-Language capabilities",
    version="1.0.0",
)

# Get CORS settings from environment
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins != "*":
    allowed_origins = [origin.strip() for origin in allowed_origins.split(",")]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
vlm = None
classes = None
last_processed_image = None  # Store the last processed image
general_qa_model = None  # Initialize the general_qa_model variable

# Load model
def load_model():
    try:
        logger.info("Loading PyTorch model...")
        
        # Get model path from environment or use default
        model_path = os.getenv("MODEL_PATH", "models/breed_model/mobilenetv2_dogbreeds.pth")
        
        # Load PyTorch model
        pytorch_model = models.mobilenet_v2(pretrained=False)
        pytorch_model.classifier[1] = nn.Linear(pytorch_model.last_channel, 120)  # 120 breeds
        
        # Load trained weights
        model_path = Path(model_path)
        logger.info(f"Loading model from: {model_path}")
        pytorch_model.load_state_dict(torch.load(model_path))
        pytorch_model.eval()
        
        # Set model to float32 explicitly
        pytorch_model = pytorch_model.float()
        
        logger.info("PyTorch model loaded successfully")
        return pytorch_model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    global model, general_qa_model
    model = load_model()
    initialize_general_qa_model()
    logger.info("Application startup complete")

# Initialize the VLM when needed
def get_vlm():
    global vlm
    if vlm is None:
        logger.info("Initializing VLM...")
        vlm = DogBreedVLM()
        logger.info("VLM initialized")
    return vlm

# Initialize the general QA model
def initialize_general_qa_model():
    global general_qa_model
    if general_qa_model is None:
        logger.info("Initializing general QA model...")
        general_qa_model = get_qa_model()
        logger.info("General QA model initialized")
    return general_qa_model

# Function to get the last processed image for VLM
def get_image_for_prediction():
    """Return the last processed image for use with visual queries."""
    global last_processed_image
    return last_processed_image

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess an image for the model."""
    global last_processed_image
    last_processed_image = image.copy()  # Store a copy of the image for later use
    
    # Apply the transformations to the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations and add batch dimension
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor

def post_process_results(output: torch.Tensor) -> list:
    """Post-process model output to get breed predictions."""
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Get breed names
    breeds = get_all_breeds()
    results = []
    
    for idx, prob in zip(top5_indices.tolist(), top5_prob.tolist()):
        # Ensure index is within bounds
        if idx < len(breeds):
            breed = breeds[idx]
            breed_info = get_breed_info(breed)
            # Convert to standard Python float
            confidence = float(prob)
            results.append({
                "breed": breed,
                "confidence": confidence,
                "info": breed_info
            })
    
    return results

@app.post("/api/classification/predict")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Ensure image is in RGB format (important for WebP images)
        if image.format == 'WEBP' or image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Get prediction with PyTorch model
        with torch.no_grad():
            output = model(input_tensor)
        
        # Post-process results
        predictions = post_process_results(output)
        
        return {
            "success": True,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/vision-language/analyze")
async def analyze_with_vlm(
    file: UploadFile = File(...),
    include_description: bool = Form(False)
):
    """Analyze an image using the VLM."""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Ensure image is in RGB format (important for WebP images)
        if image.format == 'WEBP' or image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get the VLM
        vlm_instance = get_vlm()
        
        # Process with VLM
        predictions = vlm_instance.process_image(image)
        
        # Get visual analysis details
        visual_details = vlm_instance.analyze_visual_details(image)
        
        # If description not needed, remove it
        if not include_description:
            for pred in predictions:
                if "description" in pred:
                    del pred["description"]
        
        # Return a combined response
        return {
            "predictions": predictions,
            "top_breed": predictions[0]["breed"] if predictions else None,
            "confidence": predictions[0]["confidence"] if predictions else None,
            "caption": visual_details.get("caption", ""),
            "colors": visual_details.get("colors", ""),
            "detailed_appearance": visual_details.get("detailed_appearance", "")
        }
     except Exception as e:
        logger.error(f"Error in VLM analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e)
        }

@app.post("/api/vision-language/query")
async def query_about_image(file: UploadFile = File(...), query: str = Form(...)):
    """Process a natural language query about the image."""
    try:
        # Log the received query for debugging
        logging.info(f"Received visual query: {query}")
        
        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        # Ensure image is in RGB format (important for WebP images)
        if image.format == 'WEBP' or image.mode != 'RGB':
            image = image.convert('RGB')
        
        start_time = time.time()  # Start timing
        
        # Make sure we have the general QA model
        general_qa = initialize_general_qa_model()
        
        # Get VLM instance first since we'll need it for visual question detection
        vlm_instance = get_vlm()
        
        # First check if it's a visual question (direct image-related query)
        is_visual_question = vlm_instance._is_visual_question(query)
        logger.info(f"Is visual question: {is_visual_question}")
        
        # If it's a visual question, process directly with BLIP
        if is_visual_question:
            logger.info(f"Processing as visual question: {query}")
            try:
                # Get response directly from visual query processing
                response_text = vlm_instance.answer_visual_query(query, image)
                
                response = {
                    "query": query,
                    "response": response_text,
                    "is_visual_question": True,
                    "processing_time": f"{time.time() - start_time:.2f}"
                }
                
                # Log the generated response for debugging
                logging.info(f"Generated response: {response_text}")
                
                return response
            except Exception as visual_error:
                # Log specific error with visual processing
                logger.error(f"Error in visual question processing: {str(visual_error)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Error processing visual question: {str(visual_error)}")
        
        # If not visual, check if it's a general dog question
        is_general_question = False
        if general_qa is not None:
            is_general_question = general_qa.is_general_dog_question(query)
        
        if is_general_question:
            # If it's a general question, route to the general QA model
            logger.info(f"Processing as general question: {query}")
            response = general_qa.answer_question(query)
            return {
                "query": query,
                "response": response,
                "is_general_question": True,
                "processing_time": f"{time.time() - start_time:.2f}"
            }
            
        # If it's neither a visual question nor a general question,
        # treat it as a breed-specific question
        logger.info(f"Processing as breed question: {query}")
        response_text = vlm_instance.answer_query(query, image)
        response = {
            "query": query,
            "response": response_text,
            "processing_time": f"{time.time() - start_time:.2f}"
        }
        
        # Log the generated response for debugging
        logging.info(f"Generated response: {response_text}")
        
        return response
        except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visual-analysis")
async def analyze_visuals(
    file: UploadFile = File(...),
    full_analysis: bool = Form(False)
):
    """Analyze the visual aspects of an image using the integrated VLM."""
    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Ensure image is in RGB format (important for WebP images)
        if image.format == 'WEBP' or image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get the VLM
        vlm_instance = get_vlm()
        
        # Get visual details analysis
        results = vlm_instance.analyze_visual_details(image)
        
        # Get predictions in case they're needed for context
        try:
            predictions = vlm_instance.process_image(image)
            top_breed = predictions[0]["breed"] if predictions else "unknown"
            results["top_breed"] = top_breed
        except Exception as pred_error:
            logger.error(f"Could not get predictions during visual analysis: {str(pred_error)}")
        
        return results
            except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a dedicated endpoint for general dog questions (no image required)
@app.post("/api/general-qa/query")
async def general_dog_question(query: str = Form(...)):
    """Answer general questions about dogs using a lightweight QA model."""
    try:
        qa_model = initialize_general_qa_model()
        response = qa_model.answer_question(query)
        
        return {
            "success": True,
            "query": query,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Error processing general question: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/breeds")
async def list_breeds():
    """List all supported dog breeds."""
    breeds = get_all_breeds()
    return {
        "breeds": breeds,
        "count": len(breeds)
    }

@app.get("/api/breeds/{breed_name}")
async def get_breed(breed_name: str):
    """Get detailed information about a specific breed."""
    try:
        info = get_breed_info(breed_name)
        return info
    except Exception as e:
        return {
            "error": f"Breed '{breed_name}' not found"
        }

@app.get("/api/system/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vlm_available": get_vlm() is not None
    }

@app.post("/api/vision-language/reasoning")
async def visual_reasoning(file: UploadFile = File(...)):
    """Provide detailed visual reasoning about dog breed identification.
    This showcases more advanced VLM capabilities like:
    - Visual feature analysis
    - Comparative reasoning between predictions
    - Confidence level explanation
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Ensure image is in RGB format (important for WebP images)
        if image.format == 'WEBP' or image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process with VLM
        predictions = get_vlm().process_image(image)
        
        # Get comparative reasoning between top predictions
        comparative_reasoning = get_vlm().visual_reasoning(predictions)
        
        # Extract the visual features that led to this identification
        visual_features = []
        if predictions[0]["breed"] in get_vlm().breed_features:
            visual_features = get_vlm().breed_features[predictions[0]["breed"]]
        
        return {
            "success": True,
            "top_breed": predictions[0]["breed"],
            "confidence": predictions[0]["confidence"],
            "confidence_statement": predictions[0]["confidence_statement"],
            "visual_reasoning": predictions[0]["visual_reasoning"],
            "comparative_reasoning": comparative_reasoning,
            "key_visual_features": visual_features,
            "predictions": [
                {
                    "breed": p["breed"],
                    "confidence": p["confidence"]
                } for p in predictions[:3]  # Return top 3 predictions
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in visual reasoning: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "healthy"} 