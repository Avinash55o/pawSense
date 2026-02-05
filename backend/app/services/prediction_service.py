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
        
        Args:
            raw_predictions: Raw predictions from classifier
            
        Returns:
            Formatted predictions with breed info
        """
        from app.utils.breed_info import get_breed_info
        
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
