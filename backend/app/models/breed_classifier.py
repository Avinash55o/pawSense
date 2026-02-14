import logging
import gc

logger = logging.getLogger(__name__)

class BreedClassifier:
    def __init__(self):
        self.model = None
        
    def initialize(self):
        """Lazy load the pretrained model"""
        if self.model is None:
            logger.info("Loading pretrained breed classifier (MobileNet)...")
            
            # Force garbage collection to free up memory before loading
            gc.collect()
            
            try:
                from transformers import pipeline
                self.model = pipeline(
                    "image-classification",
                    model="google/mobilenet_v2_1.0_224",
                    device="cpu"
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def predict(self, image):
        """Predict breed from image"""
        if self.model is None:
            self.initialize()
        
        try:
            results = self.model(image, top_k=5)
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return empty list or basic fallback rather than crashing
            return []

# Singleton instance
_classifier = BreedClassifier()

def get_classifier():
    return _classifier