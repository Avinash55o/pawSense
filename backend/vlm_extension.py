"""
VLM (Vision-Language Model) Extension using Google Gemini API
Provides visual question answering and enhanced breed analysis with natural language descriptions.
"""
import logging
import os
from typing import Dict, List, Any, Optional
from PIL import Image
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DogBreedVLM:
    """
    A Vision-Language Model interface using Google Gemini API (FREE tier).
    Provides visual question answering and enhanced breed descriptions.
    """
    
    def __init__(self, classifier=None):
        """Initialize the VLM with the breed classifier and Google Gemini API."""
        self.classifier = classifier
        
        # Load language templates and features for fallback
        self.templates = self._load_language_templates()
        self.breed_features = self._load_breed_features()
        self.breeds = self._load_all_breeds_list()
        
        # Configure Google Gemini API
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_GEMINI_API_KEY not found. VQA will use template responses only.")
            self.gemini_model = None
        else:
            try:
                genai.configure(api_key=api_key)
                # Use Gemini 2.5 Flash (free tier, fast)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("Google Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
                self.gemini_model = None

    def _query_gemini_api(self, image: Image.Image, question: str) -> Optional[str]:
        """Send a visual question to Google Gemini API."""
        if not self.gemini_model:
            return "Gemini API not available. Please set GOOGLE_GEMINI_API_KEY in environment."

        try:
            # Generate response
            response = self.gemini_model.generate_content([question, image])
            
            if response and response.text:
                answer = response.text.strip()
                logger.info(f"Gemini VQA: Q='{question[:50]}...' A='{answer[:100]}...'")
                return answer
            else:
                logger.warning("Gemini returned empty response")
                return "Unable to analyze the image at this time."
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini API request failed: {repr(e)}")
            
            # Handle specific error types with user-friendly messages
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                return "⚠️ This feature is temporarily unavailable due to API rate limits. Please try again later or use the Basic analysis mode."
            elif "400" in error_msg:
                return "Unable to process this image. Please try a different image."
            elif "401" in error_msg or "403" in error_msg or "API key" in error_msg:
                return "API authentication issue. Please contact support."
            else:
                return f"Unable to analyze the image at this time. Please use Basic analysis mode."

    def process_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Process an image using the prediction service (with filtering)."""
        try:
            # Use the prediction service which filters out non-dog classifications
            from app.services.prediction_service import get_prediction_service
            
            prediction_service = get_prediction_service()
            if prediction_service is None:
                return [{
                    "breed": "unknown",
                    "confidence": 0.0,
                    "info": {"description": "Prediction service not available"}
                }]
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Get filtered predictions from prediction service
            predictions = prediction_service.predict_breed(image)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return [{
                "breed": "error",
                "confidence": 0.0,
                "info": {"description": f"Processing error: {str(e)}"}
            }]

    def analyze_with_description(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Analyze image and add natural language descriptions.
        Uses Gemini API if available, otherwise falls back to templates.
        """
        # Get basic predictions from prediction service (with filtering)
        predictions = self.process_image(image)
        
        if not predictions:
            return []
        
        # Enhance top prediction with Gemini description (if available)
        top_breed = predictions[0]['breed']
        breed_info = predictions[0].get('info', {})
        
        # Generate description using Gemini (if available)
        if self.gemini_model and top_breed != "unknown":
            try:
                description_prompt = f"Describe a {breed_info.get('name', top_breed)} dog breed in 2-3 sentences. Focus on temperament and characteristics."
                gemini_desc = self._query_gemini_api(image, description_prompt)
                if gemini_desc and not gemini_desc.startswith("Error"):
                    predictions[0]['description'] = gemini_desc
                else:
                    # Fallback to template
                    predictions[0]['description'] = self._get_template_description(top_breed, breed_info)
            except Exception as e:
                logger.warning(f"Gemini description failed, using template: {e}")
                predictions[0]['description'] = self._get_template_description(top_breed, breed_info)
        else:
            # Use template description
            predictions[0]['description'] = self._get_template_description(top_breed, breed_info)
        
        # Add visual reasoning
        predictions[0]['visual_reasoning'] = self._generate_visual_reasoning(top_breed, breed_info)
        predictions[0]['confidence_statement'] = self._get_confidence_statement(predictions[0]['confidence'])
        
        # Enhance other predictions with templates
        for i in range(1, len(predictions)):
            breed_name = predictions[i]['breed']
            info = predictions[i].get('info', {})
            predictions[i]['description'] = self._get_template_description(breed_name, info)
            predictions[i]['visual_reasoning'] = self._generate_visual_reasoning(breed_name, info)
            predictions[i]['confidence_statement'] = self._get_confidence_statement(predictions[i]['confidence'])
        
        return predictions

    def visual_reasoning_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform deep visual reasoning analysis using Gemini.
        Falls back to template-based analysis if Gemini unavailable or errors occur.
        """
        # Get predictions
        predictions = self.analyze_with_description(image)
        
        if not predictions or len(predictions) < 1:
            return {"error": "Unable to analyze image", "success": False}
        
        top_breed = predictions[0]['breed']
        second_breed = predictions[1]['breed'] if len(predictions) > 1 else None
        top_info = predictions[0].get('info', {})       
        # Try Gemini for visual reasoning (if available)
        visual_reasoning = None
        comparative_reasoning = None
        key_features = []
        
        if self.gemini_model:
            try:
                # Visual reasoning
                reasoning_prompt = f"Looking at this dog image, what visual features indicate it's a {top_info.get('name', top_breed)}? List 3-5 specific physical characteristics you can see."
                visual_reasoning = self._query_gemini_api(image, reasoning_prompt)
                
                # Check if we got an error message back
                if visual_reasoning and ("⚠️" in visual_reasoning or "Unable to" in visual_reasoning or "unavailable" in visual_reasoning):
                    # Gemini failed, use template
                    logger.warning("Gemini returned error, falling back to template for visual reasoning")
                    visual_reasoning = None  # Will trigger fallback below
                
                # Comparative reasoning
                if second_breed and second_breed != "unknown":
                    second_info = predictions[1].get('info', {})
                    compare_prompt = f"Why does this dog look more like a {top_info.get('name', top_breed)} than a {second_info.get('name', second_breed)}? Explain in one sentence."
                    comparative_reasoning = self._query_gemini_api(image, compare_prompt)
                    
                    # Check for error
                    if comparative_reasoning and ("⚠️" in comparative_reasoning or "Unable to" in comparative_reasoning):
                        comparative_reasoning = None
                
                # Extract features from Gemini
                features_prompt = "List the dog's key physical features (ears type, coat type, build, color) as short phrases, comma-separated."
                features_response = self._query_gemini_api(image, features_prompt)
                if features_response and not ("⚠️" in features_response or "Unable to" in features_response or "Error" in features_response):
                    key_features = [f.strip() for f in features_response.split(',')[:5]]
                    
            except Exception as e:
                logger.warning(f"Gemini reasoning failed, using templates: {e}")
        
        # Fallback to templates if Gemini failed or unavailable
        if not visual_reasoning:
            visual_reasoning = self._generate_visual_reasoning(top_breed, top_info)
        
        if not comparative_reasoning and second_breed and second_breed != "unknown":
            second_info = predictions[1].get('info', {})
            comparative_reasoning = f"High confidence in primary prediction."
        
        if not key_features:
            key_features = self._extract_breed_features(top_breed)
        
        return {
            "success": True,
            "top_breed": top_breed,
            "confidence": predictions[0]['confidence'],
            "visual_reasoning": visual_reasoning,
            "comparative_reasoning": comparative_reasoning or "High confidence in primary prediction.",
            "key_visual_features": key_features,
            "predictions": predictions
        }

    def answer_visual_query(self, query: str, image: Image.Image) -> str:
        """
        Answer a specific question about the image using Gemini API.
        This is the interactive Q&A feature.
        """
        if not self.gemini_model:
            return "Visual Q&A is not available. Please configure GOOGLE_GEMINI_API_KEY."
        
        try:
            # Use Gemini to answer the question
            answer = self._query_gemini_api(image, query)
            return answer if answer else "Unable to answer at this time."
            
        except Exception as e:
            logger.error(f"Visual query failed: {e}")
            return f"Error: {str(e)[:100]}"

    # ========================================================================
    # Helper Methods (Templates & Fallbacks)
    # ========================================================================

    def _get_template_description(self, breed_name: str, breed_info: Dict) -> str:
        """Generate template-based description."""
        name = breed_info.get('name', breed_name.replace('_', ' ').title())
        desc = breed_info.get('description', f"The {name} is a distinctive breed with unique characteristics and temperament.")
        characteristics = breed_info.get('characteristics', [])
        size = breed_info.get('size', 'medium')
        
        char_str = " and ".join(characteristics[:2]) if characteristics else "distinctive"
        return f"The image shows a {name}. {desc} They are generally {size.lower()} dogs."

    def _generate_visual_reasoning(self, breed_name: str, breed_info: Dict) -> str:
        """Generate visual reasoning from breed features."""
        features = self.breed_features.get(breed_name, [])
        name = breed_info.get('name', breed_name.replace('_', ' ').title())
        
        if features and len(features) >= 2:
            feat_str = ", ".join(features[:3])
            return f"The {feat_str} are characteristic of the {name} breed."
        
        appearance = breed_info.get('appearance', '')
        if appearance:
            key_traits = appearance.split(',')[:2]
            return f"The {', '.join(key_traits)} are characteristic of the {name} breed."
        
        return f"Based on overall appearance and structure, this matches the {name} breed profile."

    def _get_confidence_statement(self, confidence: float) -> str:
        """Generate confidence statement."""
        if confidence > 0.7:
            return "I'm very confident this is a"
        elif confidence > 0.4:
            return "This appears to be a"
        else:
            return "This might be a"

    def _extract_breed_features(self, breed_name: str) -> List[str]:
        """Extract key features for a breed."""
        return self.breed_features.get(breed_name, [
            "distinctive coat",
            "breed-typical build",
            "characteristic features"
        ])[:5]

    def _get_breed_info(self, breed_name: str) -> Dict[str, Any]:
        """Get breed information from templates."""
        from app.utils.breed_info import get_breed_info
        return get_breed_info(breed_name)

    def _load_language_templates(self) -> Dict:
        """Load language templates for descriptions."""
        return {
            "greeting": "Based on the image analysis:",
            "confidence_high": "I'm quite confident this is",
            "confidence_medium": "This appears to be",
            "confidence_low": "This might be"
        }

    def _load_breed_features(self) -> Dict[str, List[str]]:
        """Load visual features for each breed."""
        # This is a subset - add more as needed
        return {
            "golden_retriever": ["golden coat", "friendly face", "floppy ears", "sturdy build"],
            "labrador_retriever": ["short dense coat", "otter tail", "broad head"],
            "german_shepherd": ["pointed ears", "black and tan coat", "athletic build"],
            "bulldog": ["wrinkled face", "stocky build", "short snout"],
            "poodle": ["curly coat", "elegant build", "long ears"],
            "beagle": ["tricolor coat", "floppy ears", "compact build"],
            "rottweiler": ["black and tan coat", "muscular build", "broad chest"],
            "yorkshire_terrier": ["silky coat", "small size", "pointed ears"],
            "boxer": ["square muzzle", "muscular build", "short coat"],
            "dachshund": ["long body", "short legs", "elongated snout"]
        }

    def _load_all_breeds_list(self) -> List[str]:
        """Load list of all supported breeds."""
        from app.utils.breed_info import get_all_breeds
        return get_all_breeds()