"""
Prediction service for dog breed classification.
"""
from PIL import Image
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling breed predictions."""
    
    def __init__(self, classifier):
        """Initialize with a breed classifier."""
        self.classifier = classifier
        self.last_processed_image = None
    
    def predict_breed(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Predict dog breed from an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            List of predictions with breed, confidence, and info
        """
        try:
            # Store image for later use (VLM queries)
            self.last_processed_image = image.copy()
            
            # Get predictions from classifier
            raw_predictions = self.classifier.predict(image)
            
            # Format predictions
            predictions = self._format_predictions(raw_predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in breed prediction: {str(e)}")
            raise
    
    def _format_predictions(self, raw_predictions: List[Dict]) -> List[Dict[str, Any]]:
        """
        Format raw predictions into standardized format.
        Only includes actual dog breeds, filtering out non-dog ImageNet classes.
        
        Args:
            raw_predictions: Raw predictions from classifier
            
        Returns:
            Formatted predictions with breed info
        """
        from app.utils.breed_info import get_breed_info, get_all_breeds
        
        # Get list of valid dog breeds
        valid_breeds = set(get_all_breeds())
        
        predictions = []
        
        for pred in raw_predictions:
            breed_name = pred['label'].lower().replace(' ', '_').replace('-', '_')
            
            # FILTER: Only include predictions that are actual dog breeds
            if breed_name not in valid_breeds:
                logger.info(f"Skipping non-dog classification: {breed_name} (score: {pred['score']:.3f})")
                continue
            
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
        
        # If no valid dog breeds found, return "unknown" placeholder
        if not predictions:
            logger.warning("No dog breeds detected in image - all top predictions were non-dog classes")
            predictions = [{
                "breed": "unknown",
                "confidence": 0.0,
                "info": {
                    "name": "Unknown",
                    "description": "Could not identify a dog breed in this image. This might not be a dog photo, or the image quality may be too low for accurate classification.",
                    "characteristics": [],
                    "size": "Unknown",
                    "energy_level": "Unknown",
                    "good_with_children": None
                }
            }]
        
        return predictions
    
    def get_last_image(self) -> Image.Image:
        """Get the last processed image."""
        return self.last_processed_image


# Singleton instance
_prediction_service = None


def get_prediction_service(classifier=None):
    """Get or create prediction service instance."""
    global _prediction_service
    
    if _prediction_service is None and classifier is not None:
        _prediction_service = PredictionService(classifier)
    
    return _prediction_service
