from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class BreedClassifier:
    def __init__(self):
        self.model = None
        
    def initialize(self):
        """Lazy load the pretrained model"""
        if self.model is None:
            logger.info("Loading pretrained breed classifier...")
            self.model = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device="cpu"
            )
            logger.info("Model loaded successfully")
    
    def predict(self, image):
        """Predict breed from image"""
        if self.model is None:
            self.initialize()
        
        results = self.model(image, top_k=5)
        return results

# Singleton instance
_classifier = BreedClassifier()

def get_classifier():
    return _classifier