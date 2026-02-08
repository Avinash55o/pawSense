"""
PawSense FastAPI Application
Main entry point for the dog breed identification API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.models.breed_classifier import get_classifier
from app.services.prediction_service import get_prediction_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="PawSense API",
    description="Dog breed identification and analysis with Vision-Language capabilities",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
breed_classifier = None
prediction_service = None
vlm = None
general_qa_model = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global breed_classifier, prediction_service, general_qa_model
    
    logger.info("Starting PawSense API...")
    
    # Initialize breed classifier
    breed_classifier = get_classifier()
    logger.info("Breed classifier ready (will load on first prediction)")
    
    # Initialize prediction service
    prediction_service = get_prediction_service(breed_classifier)
    logger.info("Prediction service initialized")
    
    # Initialize general QA model
    if settings.ENABLE_QA:
        try:
            from app.models.general_qa import get_qa_model
            # Don't force load the model here, let it be lazy loaded on first request
            # general_qa_model = get_qa_model() 
            logger.info("General QA enabled (lazy loading)")
        except Exception as e:
            logger.error(f"Failed to initialize General QA: {e}")
            settings.ENABLE_QA = False
    else:
        logger.info("General QA disabled to save memory")
        
    # Force garbage collection
    import gc
    gc.collect()
    
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down PawSense API...")


# Import and include routes
from app.api.routes import router

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }
