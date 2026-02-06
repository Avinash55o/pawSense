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
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", 
        "*"
    ).split(",") if os.getenv("ALLOWED_ORIGINS") != "*" else ["*"]
    
    # Model Settings
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/models")
    
    # Feature Flags
    ENABLE_QA: bool = os.getenv("ENABLE_QA", "true").lower() == "true"
    
    # Deployment Settings
    IS_RENDER: bool = os.getenv("RENDER", "false").lower() == "true"


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
