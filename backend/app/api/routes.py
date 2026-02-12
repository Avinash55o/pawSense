"""
API routes for PawSense application.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from PIL import Image
import io
import logging
import time

from app.services.prediction_service import get_prediction_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/api/classification/predict")
async def predict_breed(file: UploadFile = File(...)):
    """
    Predict dog breed from an uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with breed information
    """
    try:
        start_time = time.time()
        logger.info(f"Received prediction request for file: {file.filename}")
        
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Could not read image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get prediction service
        prediction_service = get_prediction_service()
        if prediction_service is None:
            raise HTTPException(status_code=503, detail="Prediction service not available")
        
        # Make prediction
        predictions = prediction_service.predict_breed(image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Prediction completed in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "predictions": predictions,
            "processing_time": f"{processing_time:.2f} seconds"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/api/breeds")
async def list_breeds():
    """Get list of all supported breeds."""
    from app.utils.breed_info import get_all_breeds
    
    breeds = get_all_breeds()
    return {
        "breeds": breeds,
        "count": len(breeds)
    }


@router.get("/api/breeds/{breed_name}")
async def get_breed_details(breed_name: str):
    """Get detailed information about a specific breed."""
    from app.utils.breed_info import get_breed_info
    
    try:
        info = get_breed_info(breed_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Breed '{breed_name}' not found")


@router.get("/api/system/health")
async def system_health():
    """System health check with service status."""
    from app.models.breed_classifier import get_classifier
    
    classifier = get_classifier()
    prediction_service = get_prediction_service()
    
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "prediction_service_ready": prediction_service is not None
    }


@router.get("/api/health")
async def health_check_alias():
    """Backward compatibility for old health check endpoint."""
    return {"status": "healthy"}


# VLM Endpoints
@router.post("/api/vision-language/analyze")
async def analyze_with_vlm(file: UploadFile = File(...)):
    """
    Analyze an image using Vision-Language Model (BLIP).
    
    Args:
        file: Uploaded image file
        
    Returns:
        VLM analysis results
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Import VLM and get classifier (lazy load)
        from vlm_extension import DogBreedVLM
        from app.main import breed_classifier
        vlm = DogBreedVLM(classifier=breed_classifier)
        
        # Process with VLM
        predictions = vlm.process_image(image)
        
        return {
            "success": True,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error in VLM analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VLM analysis failed: {str(e)}")


@router.post("/api/vision-language/reasoning")
async def visual_reasoning(file: UploadFile = File(...)):
    """
    Perform visual reasoning on an image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Visual reasoning results
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Import VLM and get classifier
        from vlm_extension import DogBreedVLM
        from app.main import breed_classifier
        vlm = DogBreedVLM(classifier=breed_classifier)
        
        # Use the correct method: visual_reasoning_analysis
        result = vlm.visual_reasoning_analysis(image)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in visual reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visual reasoning failed: {str(e)}")


@router.post("/api/vision-language/query")
async def visual_query(file: UploadFile = File(...), query: str = Form("")):
    """
    Query about an image using VLM.
    
    Args:
        file: Uploaded image file
        query: Question about the image
        
    Returns:
        Answer to the query
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Import VLM and get classifier
        from vlm_extension import DogBreedVLM
        from app.main import breed_classifier
        vlm = DogBreedVLM(classifier=breed_classifier)
        
        # Answer query (correct parameter order: query first, then image)
        answer = vlm.answer_visual_query(query, image)
        
        return {
            "success": True,
            "query": query,
            "answer": answer,
            "response": answer  # Add 'response' field for frontend compatibility
        }
        
    except Exception as e:
        logger.error(f"Error in visual query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visual query failed: {str(e)}")


# General QA Endpoint
@router.post("/api/general-qa/query")
async def general_qa_query(query: str = Form("")):
    """
    Answer general questions about dogs.
    
    Args:
        query: Question about dogs (from form data)
        
    Returns:
        Answer to the query
    """
    try:
        return {
            "success": True,
            "query": query,
            "answer": "General dog Q&A is currently simplified. Please upload an image to ask specific questions about a dog!",
            "response": "General dog Q&A is currently simplified. Please upload an image to ask specific questions about a dog!" 
        }
        
    except Exception as e:
        logger.error(f"Error in general QA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"General QA failed: {str(e)}")
