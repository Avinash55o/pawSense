"""
Configuration settings for PawSense backend application.
"""
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings and configuration."""
    
    # API Settings
    APP_NAME: str = "PawSense API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Dog breed identification and analysis with Vision-Language capabilities"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", 
        "*"
    ).split(",") if os.getenv("ALLOWED_ORIGINS") != "*" else ["*"]
    
    # Model Settings
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/models")
    BREED_CLASSIFIER_MODEL: str = os.getenv("BREED_CLASSIFIER_MODEL", "microsoft/resnet-50")
    BLIP_MODEL: str = os.getenv("BLIP_MODEL", "Salesforce/blip-vqa-base")
    QA_MODEL: str = os.getenv("QA_MODEL", "distilbert-base-cased-distilled-squad")
    
    # Feature Flags
    ENABLE_BLIP: bool = os.getenv("ENABLE_BLIP", "true").lower() == "true"
    ENABLE_QA: bool = os.getenv("ENABLE_QA", "true").lower() == "true"
    
    # Upload Settings
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    
    # Performance Settings
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "120"))
    
    # Deployment Settings
    IS_RENDER: bool = os.getenv("RENDER", "false").lower() == "true"
    IS_PRODUCTION: bool = os.getenv("NODE_ENV", "development") == "production"


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
