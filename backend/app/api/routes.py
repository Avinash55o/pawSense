"""
API routes for PawSense application.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
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
