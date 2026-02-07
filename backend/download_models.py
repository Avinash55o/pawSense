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
    
    # 1. Download MobileNetV2 for breed classification
    logger.info("\n1. Downloading MobileNetV2 (breed classification)...")
    try:
        pipeline("image-classification", model="google/mobilenet_v2_1.0_224", device="cpu")
        logger.info("✅ MobileNetV2 downloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to download MobileNetV2: {e}")
    
    # 2. Download DistilBERT for general Q&A
    logger.info("\n2. Downloading DistilBERT (general Q&A)...")
    try:
        AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
        logger.info("✅ DistilBERT downloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to download DistilBERT: {e}")

    # 3. BLIP (commented out to save resources on free tier, enable if VLM is used)
    # logger.info("\n3. Downloading BLIP (visual Q&A)...")
    # try:
    #     BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    #     BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    #     logger.info("✅ BLIP downloaded successfully")
    # except Exception as e:
    #     logger.error(f"❌ Failed to download BLIP: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All models downloaded successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    download_models()
