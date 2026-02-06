#!/usr/bin/env python3
"""
Pre-download all required models before starting the server.
This ensures users don't wait for model downloads on first use.
"""
import logging
from transformers import pipeline, BlipProcessor, BlipForQuestionAnswering, AutoTokenizer, AutoModelForQuestionAnswering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download all required models."""
    
    logger.info("=" * 60)
    logger.info("Pre-downloading all models...")
    logger.info("=" * 60)
    
    # 1. Download ResNet-50 for breed classification
    logger.info("\n1. Downloading ResNet-50 (breed classification)...")
    try:
        pipeline("image-classification", model="microsoft/resnet-50", device="cpu")
        logger.info("✅ ResNet-50 downloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to download ResNet-50: {e}")
    
    # 2. Download BLIP for visual Q&A
    logger.info("\n2. Downloading BLIP (visual Q&A)...")
    try:
        BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        logger.info("✅ BLIP downloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to download BLIP: {e}")
    
    # 3. Download DistilBERT for general Q&A
    logger.info("\n3. Downloading DistilBERT (general Q&A)...")
    try:
        AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
        logger.info("✅ DistilBERT downloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to download DistilBERT: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All models downloaded successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    download_models()
