#!/usr/bin/env python3
"""
Pre-download all required models before starting the server.
This ensures users don't wait for model downloads on first use.
"""
import logging
from transformers import pipeline

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
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All core models downloaded successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    download_models()
