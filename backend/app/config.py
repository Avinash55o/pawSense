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


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
