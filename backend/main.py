from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from pathlib import Path
import logging
import traceback
from breed_info import get_breed_info, get_all_breeds
from typing import Optional, List, Dict, Any, Union
import time
import re
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our VLM extension
from vlm_extension import DogBreedVLM
# Import our general QA model
from general_qa import get_qa_model
# Import our new pretrained breed classifier
from models.breed_classifier import get_classifier

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
breed_classifier = None  # Pretrained breed classifier
vlm = None
classes = None
last_processed_image = None  # Store the last processed image
general_qa_model = None  # Initialize the general_qa_model variable

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    global breed_classifier, general_qa_model
    # Get the pretrained classifier instance
    breed_classifier = get_classifier()
    # Initialize it lazily (will load on first use)
    logger.info("Breed classifier ready (will load on first prediction)")
    
    # Initialize general QA model
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


@app.post("/api/classification/predict")
async def analyze_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Received image analysis request for file: {file.filename}")
        start_time = time.time()
        
        # Check if classifier is available
        if breed_classifier is None:
            logger.error("Breed classifier not loaded. Cannot process image.")
            return {
                "success": False,
                "error": "Model not available at this time. Please try again later."
            }
            
        # Read image with proper error handling
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception as img_error:
            logger.error(f"Error reading uploaded image: {str(img_error)}")
            return {
                "success": False,
                "error": f"Could not read uploaded image: {str(img_error)}"
            }
        
        # Ensure image is in RGB format (important for WebP images)
        try:
            if image.format == 'WEBP' or image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as convert_error:
            logger.error(f"Error converting image to RGB: {str(convert_error)}")
            return {
                "success": False,
                "error": f"Could not process image format: {str(convert_error)}"
            }
        
        # Store the image for later use (for VLM queries)
        global last_processed_image
        last_processed_image = image.copy()
        
        # Get prediction with pretrained classifier
        try:
            # The HuggingFace pipeline handles preprocessing internally
            raw_predictions = breed_classifier.predict(image)
            
            # Convert HuggingFace output to our format
            predictions = []
            for pred in raw_predictions:
                breed_name = pred['label'].lower().replace(' ', '_')
                # Try to get breed info, fallback to generic if not found
                try:
                    breed_info = get_breed_info(breed_name)
                except:
                    breed_info = {
                        "name": pred['label'],
                        "description": f"A {pred['label']} dog breed.",
                        "characteristics": ["Loyal", "Intelligent"],
                        "size": "Medium",
                        "energy_level": "Medium",
                        "good_with_children": True
                    }
                
                predictions.append({
                    "breed": breed_name,
                    "confidence": float(pred['score']),
                    "info": breed_info
                })
                
        except Exception as model_error:
            logger.error(f"Error during model inference: {str(model_error)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Error analyzing image with model: {str(model_error)}"
            }
        
        # Log success and timing
        processing_time = time.time() - start_time
        logger.info(f"Image analyzed successfully in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "predictions": predictions,
            "processing_time": f"{processing_time:.2f} seconds"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
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
        "model_loaded": breed_classifier is not None,
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

@app.get("/api/diagnostics")
async def run_diagnostics():
    """Diagnostic endpoint to check model loading status and system resources."""
    import gc
    import psutil
    import sys
    import platform
    
    try:
        process = psutil.Process()
        
        # Get model status
        model_status = {
            "main_model_loaded": breed_classifier is not None,
            "vlm_initialized": vlm is not None if 'vlm' in globals() else False,
            "qa_model_initialized": general_qa_model is not None if 'general_qa_model' in globals() else False,
        }
        
        # Get memory info
        memory_info = {
            "process_memory_mb": process.memory_info().rss / (1024 * 1024),
            "available_system_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
            "memory_percent": psutil.virtual_memory().percent,
        }
        
        # Python info
        python_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_available": "Yes" if 'torch' in sys.modules else "No",
            "torch_version": torch.__version__ if 'torch' in sys.modules else "Not loaded",
            "transformers_version": sys.modules.get('transformers').__version__ if 'transformers' in sys.modules else "Not loaded",
            "gc_count": gc.get_count(),
        }
        
        # Try to check GPU info if available
        gpu_info = {}
        try:
            if 'torch' in sys.modules and torch.cuda.is_available():
                gpu_info = {
                    "cuda_available": True,
                    "cuda_version": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                    "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                }
            else:
                gpu_info = {"cuda_available": False}
        except Exception as gpu_error:
            gpu_info = {"cuda_available": False, "error": str(gpu_error)}
        
        # Environment variables (redacted for security)
        env_vars = {
            "PORT": os.getenv("PORT", "Not set"),
            "ALLOWED_ORIGINS": os.getenv("ALLOWED_ORIGINS", "Not set"),
            "MODEL_PATH": os.getenv("MODEL_PATH", "Not set"),
            "RENDER": os.getenv("RENDER", "Not set"),
        }
        
        return {
            "status": "ok",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_status": model_status,
            "memory_info": memory_info,
            "python_info": python_info,
            "gpu_info": gpu_info,
            "environment": env_vars,
        }
    except Exception as e:
        logger.error(f"Error in diagnostics: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        } 