import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deployment_config")

# Deployment environment detection
def is_render_env():
    """Check if we're running in Render."""
    return os.environ.get('RENDER', '') == 'true'

# Memory-constrained environment settings
def get_deployment_settings():
    """Get settings optimized for deployment environments."""
    settings = {
        # BLIP model configurations
        "use_cpu_for_blip": True,                     # Force CPU for BLIP to reduce memory usage
        "blip_model": "Salesforce/blip-vqa-base",     # Use base model instead of large
        "blip_max_length": 50,                        # Reduce max token length for responses
        
        # General settings
        "debug_mode": is_render_env(),                # Enable extra logging in production
        "lazy_load_models": True,                     # Only load models when needed
        
        # Memory management
        "release_memory": True,                       # Release memory after processing
        
        # Timeout settings
        "model_load_timeout": 30,                     # Timeout for model loading in seconds
        
        # Fallbacks
        "enable_fallbacks": True,                     # Enable fallback responses if model fails
    }
    
    # Log the current configuration
    logger.info(f"Deployment settings: {settings}")
    return settings 