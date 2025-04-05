import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import re
from breed_info import get_breed_info, get_all_breeds
import random
import logging
import traceback
import time
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image

try:
    from transformers import BlipProcessor, BlipForQuestionAnswering
except ImportError:
    logging.warning("BLIP transformers modules not found. Visual QA will not work.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DogBreedVLM:
    """A lightweight Vision-Language Model for dog breed identification and description."""
    
    def __init__(self, vision_model_path='models/breed_model/mobilenetv2_dogbreeds.pth'):
        """Initialize the VLM with the vision model and language templates."""
        self.vision_model = self._load_vision_model(vision_model_path)
        self.templates = self._load_language_templates()
        self.breeds = get_all_breeds()
        self.breed_features = self._load_breed_features()
        self.blip_model = None
        self.blip_processor = None
        self.blip_initialized = False
        
    def _load_vision_model(self, model_path):
        """Load the vision model that was previously trained."""
        from main import load_model
        return load_model()
    
    def _load_blip_model(self):
        """Lazy initialization of BLIP model for visual understanding."""
        try:
            # Import deployment settings
            try:
                from deployment_config import get_deployment_settings
                settings = get_deployment_settings()
            except ImportError:
                settings = {
                    "use_cpu_for_blip": True,
                    "blip_model": "Salesforce/blip-vqa-base",
                    "blip_max_length": 100,
                    "model_load_timeout": 60,
                    "enable_fallbacks": True
                }
            
            logger.info("Loading BLIP model for enhanced visual understanding...")
            model_name = settings["blip_model"]
            
            import signal
            
            class TimeoutException(Exception):
                pass
            
            def timeout_handler(signum, frame):
                raise TimeoutException("Model loading timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            
        
            signal.alarm(settings["model_load_timeout"])
            
            try:
                import gc
                import torch
                from transformers import BlipProcessor, BlipForQuestionAnswering
                
             
                gc.collect()
                if torch.cuda.is_available() and not settings["use_cpu_for_blip"]:
                    torch.cuda.empty_cache()
                
               
                self.blip_processor = BlipProcessor.from_pretrained(model_name)
                
          
                device = "cpu" if settings["use_cpu_for_blip"] else "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading BLIP model on {device}")
                
          
                self.blip_model = BlipForQuestionAnswering.from_pretrained(
                    model_name,
                    device_map=device,
                    low_cpu_mem_usage=True
                )
                
             
                self.blip_model.eval()
                self.blip_initialized = True
                logger.info("BLIP model loaded successfully")
                
               
                signal.alarm(0)
                return True
                
            except TimeoutException:
                logger.error("BLIP model loading timed out - using fallbacks")
                if settings["enable_fallbacks"]:
                    self.blip_initialized = False
                    return False
            finally:
          
                signal.alarm(0)
                
        except Exception as e:
            logger.error(f"Error loading BLIP model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_breed_features(self):
        """Load distinctive visual features for breeds to enable visual reasoning."""
        features = {
            "golden_retriever": ["golden coat", "friendly face", "medium to large size", "floppy ears"],
            "german_shepherd": ["pointed ears", "black and tan coloring", "alert expression", "strong build"],
            "beagle": ["tri-color coat", "long droopy ears", "compact body", "short legs"],
            "pug": ["wrinkled face", "curled tail", "flat nose", "compact body"],
            "samoyed": ["fluffy white coat", "smiling expression", "pointed ears", "curled tail"],
            "bernese_mountain_dog": ["tri-color coat", "large size", "strong build", "gentle expression"],
            "shetland_sheepdog": ["collie-like appearance", "small size", "long pointed nose", "thick coat"],
            "scottish_deerhound": ["rough coat", "tall and lean body", "long legs", "gentle expression"]
        }
        
       
        for breed in self.breeds:
            if breed not in features:
                breed_name = breed.replace("_", " ")
                if "terrier" in breed_name:
                    features[breed] = ["alert expression", "compact body", "distinctive terrier coat", "energetic stance"]
                elif "hound" in breed_name:
                    features[breed] = ["long ears", "muscular body", "focused expression", "athletic build"]
                elif "spaniel" in breed_name:
                    features[breed] = ["floppy ears", "medium coat", "friendly expression", "sturdy body"]
                else:
                    features[breed] = ["distinctive breed appearance", "unique coat pattern", "characteristic stance"]
        
        return features
    
    def _load_language_templates(self):
        """Load text generation templates for breed descriptions."""
        templates = {
            "breed_description": [
                "This is a {breed}, which is known for {characteristic}. {breed_info}",
                "I can identify this as a {breed}. {breed_info} These dogs are {characteristic}.",
                "The image shows a {breed}. {breed_info} They are generally {size} dogs with {energy_level} energy levels."
            ],
            "comparison": [
                "While this looks similar to a {similar_breed}, it's actually a {breed} because of {distinguishing_feature}.",
                "This {breed} might be confused with a {similar_breed}, but you can tell by the {distinguishing_feature}.",
                "Although {similar_breed}s and {breed}s look similar, this is a {breed} based on its {distinguishing_feature}."
            ],
            "query_responses": {
                "size": "The {breed} is a {size} sized dog breed.",
                "temperament": "The {breed} is known to be {characteristics}.",
                "good_with_children": "The {breed} is {child_friendly} with children.",
                "energy": "The {breed} has a {energy_level} energy level, making it {energy_description}.",
                "appearance": "The {breed} typically has {appearance_details}.",
                "origin": "The {breed} originated from {origin}, where it was bred for {purpose}.",
                "lifespan": "The typical lifespan of a {breed} is {lifespan}."
            },
            "visual_reasoning": [
                "Looking at this image, I can identify this as a {breed} because of its {feature1} and {feature2}.",
                "The {feature1}, {feature2}, and overall {feature3} are characteristic of the {breed} breed.",
                "This dog shows the classic {breed} traits: {feature1}, {feature2}, and {feature3}."
            ],
            "general_unknown": [
                "I'm not sure about that aspect of this dog breed.",
                "That information isn't in my knowledge base for this breed.",
                "I don't have specific details about that for this breed."
            ],
            "confidence_levels": {
                "high": [
                    "I'm very confident this is a {breed}.",
                    "This is definitely a {breed}.",
                    "The distinctive features clearly identify this as a {breed}."
                ],
                "medium": [
                    "This appears to be a {breed}, though there are some similarities with other breeds.",
                    "I'm reasonably confident this is a {breed}, based on several key features.",
                    "Most likely a {breed}, though the angle makes it a bit harder to be 100% certain."
                ],
                "low": [
                    "This might be a {breed}, but I'm not entirely certain.",
                    "This could be a {breed}, though it shares features with several other breeds.",
                    "I'm seeing some {breed} characteristics, but can't be fully confident from this image."
                ]
            },
            "visual_queries": {
                "color": "Based on the image, this dog is {color_description}.",
                "pattern": "The dog's coat pattern can be described as {pattern_description}.",
                "appearance": "Visually, this dog appears to be {appearance_description}.",
                "coat": "This dog has a {coat_type} coat that is {coat_description}."
            }
        }
        
        return templates
    
    def process_image(self, image):
        """Process an image and return breed predictions with descriptions."""
        from main import preprocess_image, post_process_results
        
  
        input_tensor = preprocess_image(image)
        
     
        with torch.no_grad():
            output = self.vision_model(input_tensor)
    
        predictions = post_process_results(output)
        
       
        for pred in predictions:
            pred["description"] = self._generate_description(pred)
            pred["visual_reasoning"] = self._generate_visual_reasoning(pred)
            
    
            if pred["confidence"] > 0.7:
                confidence_level = "high"
            elif pred["confidence"] > 0.4:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            pred["confidence_statement"] = random.choice(self.templates["confidence_levels"][confidence_level]).format(
                breed=pred["breed"].replace("_", " ").title()
            )
        
        return predictions
    
    def _generate_description(self, prediction):
        """Generate a natural language description for a breed prediction."""
        import random
        
        breed = prediction["breed"].replace("_", " ").title()
        info = prediction["info"]
        
    
        template = random.choice(self.templates["breed_description"])
        
       
        characteristic = "being " + " and ".join(info["characteristics"][:2]).lower()
        
     
        breed_info = info.get("description", f"They are {info['size']} dogs with {info['energy_level'].lower()} energy levels.")
        
     
        description = template.format(
            breed=breed,
            characteristic=characteristic,
            breed_info=breed_info,
            size=info["size"].lower(),
            energy_level=info["energy_level"].lower()
        )
        
        return description
    
    def _generate_visual_reasoning(self, prediction):
        """Generate visual reasoning explanation for why this breed was identified."""
        import random
        
        breed = prediction["breed"]
        breed_name = breed.replace("_", " ").title()
        
      
        if breed in self.breed_features:
            features = self.breed_features[breed]
        else:
            
            features = ["distinctive appearance", "characteristic body structure", "typical coat pattern"]
        
       
        template = random.choice(self.templates["visual_reasoning"])
        
       
        selected_features = random.sample(features, min(3, len(features)))
        
       
        reasoning = template.format(
            breed=breed_name,
            feature1=selected_features[0],
            feature2=selected_features[1] if len(selected_features) > 1 else "overall appearance",
            feature3="appearance" if len(selected_features) < 3 else selected_features[2]
        )
        
        return reasoning
    
    def answer_query(self, query, image_or_predictions):
        """Answer a natural language query about the detected dog breeds.
        
        Args:
            query: The user's question
            image_or_predictions: Either an image or a list of predictions
        """
        query = query.lower()
        
       
        if self._is_visual_question(query):
           
            if isinstance(image_or_predictions, Image.Image):
                return self.answer_visual_query(query, image_or_predictions)
          
            else:
                from main import get_image_for_prediction 
                try:
                   
                    image = get_image_for_prediction()
                    if image:
                        return self.answer_visual_query(query, image)
                except:
                    pass  
        
       
        predictions = []
        if isinstance(image_or_predictions, Image.Image):
            predictions = self.process_image(image_or_predictions)
        else:
            predictions = image_or_predictions
        
       
        if not predictions or len(predictions) == 0:
            return "I don't see a recognizable dog breed in this image to answer your question."
        
      
        top_breed = predictions[0]
        breed_name = top_breed["breed"].replace("_", " ").title()
        
      
        info = top_breed["info"]
        
    
        if re.search(r'\b(size|how big|how large|how small)\b', query):
            response = self.templates["query_responses"]["size"].format(
                breed=breed_name, 
                size=info["size"].lower()
            )
            
        elif re.search(r'\b(temperament|personality|character|behave|behavior)\b', query):
            characteristics = ", ".join(info["characteristics"]).lower()
            response = self.templates["query_responses"]["temperament"].format(
                breed=breed_name, 
                characteristics=characteristics
            )
            
        elif re.search(r'\b(kids|children|child|family)\b', query):
            child_friendly = "generally good" if info.get("good_with_children", True) else "not always ideal"
            response = self.templates["query_responses"]["good_with_children"].format(
                breed=breed_name, 
                child_friendly=child_friendly
            )
            
        elif re.search(r'\b(energy|active|exercise|activity)\b', query):
            energy_map = {
                "High": "very active and requires plenty of exercise",
                "Medium": "moderately active and enjoys regular exercise",
                "Low": "relatively calm and doesn't require extensive exercise"
            }
            energy_description = energy_map.get(info["energy_level"], "requiring a moderate amount of exercise")
            
            response = self.templates["query_responses"]["energy"].format(
                breed=breed_name, 
                energy_level=info["energy_level"].lower(),
                energy_description=energy_description
            )
            
        elif re.search(r'\b(look|appearance|coat|color|fur|physical)\b', query):
         
            if "appearance" in info:
                appearance = info["appearance"]
            else:
                appearance = f"distinctive features common to the {breed_name} breed"
                
            response = self.templates["query_responses"]["appearance"].format(
                breed=breed_name, 
                appearance_details=appearance
            )
            
        elif re.search(r'\b(origin|history|come from|developed|bred)\b', query):
           
            if "origin" in info and "purpose" in info:
                response = self.templates["query_responses"]["origin"].format(
                    breed=breed_name,
                    origin=info["origin"],
                    purpose=info["purpose"]
                )
            else:
                response = f"I don't have detailed information about the origin of the {breed_name} in my database."
            
        elif re.search(r'\b(life|lifespan|longevity|live|age)\b', query):
            
            if "lifespan" in info:
                response = self.templates["query_responses"]["lifespan"].format(
                    breed=breed_name,
                    lifespan=info["lifespan"]
                )
            else:
                response = f"I don't have specific lifespan information for the {breed_name} in my database."
            
        elif re.search(r'\b(compare|difference|similar|vs|versus)\b', query):
           
            if len(predictions) > 1:
                similar_breed = predictions[1]["breed"].replace("_", " ").title()
                
               
                top_breed_features = self.breed_features.get(top_breed["breed"], ["distinctive appearance"])[0]
                
                response = random.choice(self.templates["comparison"]).format(
                    breed=breed_name,
                    similar_breed=similar_breed,
                    distinguishing_feature=top_breed_features
                )
            else:
                response = f"This is clearly a {breed_name} and doesn't look similar to other breeds in my database."
                
        elif re.search(r'\b(why|how can you tell|identify|recognize)\b', query):
            
            response = top_breed.get("visual_reasoning", f"I identified this as a {breed_name} based on its distinctive features.")
            
        elif re.search(r'\b(confidence|sure|certain)\b', query):
           
            response = top_breed.get("confidence_statement", f"I'm {top_breed['confidence']:.1%} confident this is a {breed_name}.")
        
        else:
            
            response = f"This appears to be a {breed_name}. " + random.choice(self.templates["general_unknown"])
        
        return response
    
    def visual_reasoning(self, predictions):
        """Provide visual reasoning for why the top prediction was chosen over alternatives."""
        if len(predictions) < 2:
            return "There's only one viable breed prediction for this image."
        
        top_breed = predictions[0]
        second_breed = predictions[1]
        
        top_name = top_breed["breed"].replace("_", " ").title()
        second_name = second_breed["breed"].replace("_", " ").title()
        
       
        if top_breed["breed"] in self.breed_features:
            top_features = self.breed_features[top_breed["breed"]][:2]
        else:
            top_features = ["distinctive appearance", "characteristic features"]
            
        top_feature_str = " and ".join(top_features)
        
        
        conf_diff = top_breed["confidence"] - second_breed["confidence"]
        conf_percent = int(conf_diff * 100)
        
        reasoning = f"I identified this dog as a {top_name} rather than a {second_name} because of its {top_feature_str}. "
        
        if conf_percent > 50:
            reasoning += f"I'm very confident in this assessment, with the {top_name} being {conf_percent}% more likely than {second_name}."
        elif conf_percent > 20:
            reasoning += f"I'm moderately confident, with the {top_name} being {conf_percent}% more likely than {second_name}."
        else:
            reasoning += f"The distinction is subtle, with the {top_name} being only {conf_percent}% more likely than {second_name}."
            
        return reasoning
    
    def _is_visual_question(self, query):
        """Determine if a query is asking about visual aspects of the image."""
        query = query.lower()
        
       
        visual_patterns = [
            r"(what|how).*(color|colour)",
            r"(what|how).*(look|appear)",
            r"describe.*dog",
            r"is.*(dog|it).*(color|colour)",
            r"is my dog",
            r"color of",
            r"appearance of"
        ]
        
        for pattern in visual_patterns:
            if re.search(pattern, query):
                return True
            
        
        visual_keywords = ["see", "look", "color", "colour", "show", "picture", "image", "photo", "spotted", "markings", "black", "white", "brown", "gray", "red"]
        for keyword in visual_keywords:
            if keyword in query:
                return True
            
        return False
    
    def answer_visual_query(self, query, image):
        """Answer a visual question directly using the BLIP model."""
      
        try:
            from deployment_config import get_deployment_settings
            settings = get_deployment_settings()
        except ImportError:
            settings = {
                "blip_max_length": 100,
                "enable_fallbacks": True
            }
            
        # Lazy load BLIP if not already initialized
        if not self.blip_initialized:
            if not self._load_blip_model():
                # BLIP failed to load - use fallback
                logger.warning("BLIP model failed to load - using visual feature fallback")
                visual_features = self._extract_visual_features(image)
                
                query_lower = query.lower()
                if 'color' in query_lower:
                    return f"This dog's coat is {visual_features.get('colors', 'varied in color')}."
                elif 'appearance' in query_lower or 'look' in query_lower:
                    return f"This dog has {visual_features.get('appearance', 'standard features')}."
                else:
                    return f"I can see a dog with {visual_features.get('colors', 'varied coloring')} and {visual_features.get('appearance', 'standard proportions')}."
        
        try:
            # Debug logging
            logger.info(f"Processing visual query: {query}")
            
            # Special handling for common visual questions
            query_lower = query.lower()
            
            # Color-related questions
            if any(pattern in query_lower for pattern in ['what color', 'what colours', 'what is the color', 'color of the dog', 'dog\'s color']):
                # First try to extract direct visual features as a backup plan
                visual_features = self._extract_visual_features(image)
                color_fallback = visual_features.get('colors', 'varied in color')
                
                # Process with BLIP model
                try:
                    # Use multiple targeted questions to get consistent results
                    color_prompts = [
                        "What color is this dog?",
                        "What is the coat color of this dog?",
                        "Describe the color of this dog's fur."
                    ]
                    
                    color_answers = []
                    for prompt in color_prompts:
                        answer = self._process_visual_query(image, prompt)
                        if answer and answer.lower() not in ["i don't know", "unknown", "i do not know"]:
                            color_answers.append(answer)
                    
                    if color_answers:
                        # Get the most informative answer
                        color_answers.sort(key=lambda x: len(x), reverse=True)
                        color_answer = color_answers[0]
                        
                        # Format response consistently
                        if not color_answer.startswith("This dog"):
                            color_answer = f"This dog's coat is {color_answer.lower()}."
                        
                        return color_answer
                    else:
                        return f"This dog's coat is {color_fallback}."
                    
                except Exception as e:
                    logger.error(f"Error in color detection: {str(e)}")
                    return f"This dog's coat is {color_fallback}."
            
            # Appearance-related questions
            elif any(pattern in query_lower for pattern in ['what does', 'appearance', 'look like', 'describe']):
                # First extract direct visual features as a backup plan
                visual_features = self._extract_visual_features(image)
                appearance_fallback = f"{visual_features.get('colors', 'colored')} with {visual_features.get('appearance', 'standard proportions')}"
                
                try:
                    # Use multiple targeted questions
                    appearance_prompts = [
                        "Describe the appearance of this dog.",
                        "What does this dog look like?",
                        "Describe the physical features of this dog."
                    ]
                    
                    appearance_answers = []
                    for prompt in appearance_prompts:
                        answer = self._process_visual_query(image, prompt)
                        if answer and answer.lower() not in ["i don't know", "unknown", "i do not know"]:
                            appearance_answers.append(answer)
                    
                    if appearance_answers:
                        # Get the most informative answer
                        appearance_answers.sort(key=lambda x: len(x), reverse=True)
                        appearance_answer = appearance_answers[0]
                        
                        # Make sure the response describes THIS dog, not the breed
                        if "typically" in appearance_answer.lower() or "breed" in appearance_answer.lower():
                            # It's describing the breed generally, not this specific dog
                            return f"In this specific image, I can see a dog with {appearance_fallback}."
                        
                        return appearance_answer
                    else:
                        return f"This dog has {appearance_fallback}."
                        
                except Exception as e:
                    logger.error(f"Error in appearance detection: {str(e)}")
                    return f"This dog has {appearance_fallback}."
            
            # Default BLIP processing for other visual questions
            try:
                inputs = self.blip_processor(image, query, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.blip_model.generate(**inputs, max_length=settings.get("blip_max_length", 100))
                answer = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                
                # If BLIP gives a non-specific or breed-related answer
                if answer.strip() in ["", ".", "?", "unknown", "not sure"] or "breed" in answer.lower():
                    # Try a direct descriptor observation
                    visual_features = self._extract_visual_features(image)
                    return f"Based on what I can see in this specific image, this dog has {visual_features.get('colors', 'varied coloring')} and {visual_features.get('appearance', 'standard proportions')}."
                    
                return answer
                
            except Exception as blip_error:
                # BLIP processing failed, use the fallback
                logger.error(f"Error in BLIP processing: {str(blip_error)}")
                logger.error(traceback.format_exc())
                
                # Extract basic visual features as fallback
                visual_features = self._extract_visual_features(image)
                return f"Based on the image, I can see that this dog has {visual_features.get('colors', 'varied coloring')} and {visual_features.get('appearance', 'standard proportions')}."
                
        except Exception as outer_error:
            logger.error(f"Error in visual query processing: {str(outer_error)}")
            logger.error(traceback.format_exc())
            
            # Last resort fallback
            try:
                visual_features = self._extract_visual_features(image)
                return f"I'm having trouble with detailed analysis, but I can see a dog with {visual_features.get('colors', 'varied coloring')}."
            except:
                return "I'm having trouble analyzing the visual details in this image."
    
    def _extract_visual_features(self, image):
        """Extract basic visual features from image using simple analysis."""
        try:
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Simple color analysis
            # Convert to HSV color space for better color analysis
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                # Extract average colors from regions of the image
                height, width, _ = img_array.shape
                
                # Calculate color distribution
                r_avg = np.mean(img_array[:, :, 0])
                g_avg = np.mean(img_array[:, :, 1])
                b_avg = np.mean(img_array[:, :, 2])
                
                # Simple color determination
                colors = []
                
                # Check for white/light colors
                if r_avg > 200 and g_avg > 200 and b_avg > 200:
                    colors.append("white")
                # Check for black/dark colors
                elif r_avg < 50 and g_avg < 50 and b_avg < 50:
                    colors.append("black")
                # Check for brown
                elif r_avg > g_avg and r_avg > b_avg and g_avg > 100:
                    colors.append("brown")
                # Check for golden/tan
                elif r_avg > 150 and g_avg > 100 and b_avg < 100:
                    colors.append("golden")
                # Check for gray
                elif abs(r_avg - g_avg) < 20 and abs(r_avg - b_avg) < 20 and abs(g_avg - b_avg) < 20:
                    colors.append("gray")
                
                # Look for secondary colors in different regions
                regions = [
                    img_array[:height//2, :width//2],  # top-left
                    img_array[:height//2, width//2:],  # top-right
                    img_array[height//2:, :width//2],  # bottom-left
                    img_array[height//2:, width//2:],  # bottom-right
                ]
                
                for i, region in enumerate(regions):
                    r_region = np.mean(region[:, :, 0])
                    g_region = np.mean(region[:, :, 1])
                    b_region = np.mean(region[:, :, 2])
                    
                    # Check for white patches
                    if r_region > 220 and g_region > 220 and b_region > 220 and "white" not in colors:
                        colors.append("white patches")
                    # Check for black patches
                    elif r_region < 50 and g_region < 50 and b_region < 50 and "black" not in colors:
                        colors.append("black patches")
                    # Check for brown patches
                    elif r_region > g_region and r_region > b_region and g_region > 100 and "brown" not in colors:
                        colors.append("brown patches")
                
                # Build color description
                color_description = " and ".join(colors[:2])
                if not color_description:
                    color_description = "varied colors"
                    
                
                appearance_features = []
                
              
                aspect_ratio = width / height
                if aspect_ratio > 1.5:
                    appearance_features.append("long body")
                elif aspect_ratio < 0.8:
                    appearance_features.append("tall standing")
                    
             
                face_region = img_array[:height//3, width//3:2*width//3]
                face_contrast = np.std(face_region)
                if face_contrast > 50:
                    appearance_features.append("distinct facial features")
                
                # Join appearance features
                appearance = ", ".join(appearance_features)
                if not appearance:
                    appearance = "standard dog proportions"
                    
                return {
                    "colors": color_description,
                    "appearance": appearance
                }
            
            return {
                "colors": "indeterminate colors",
                "appearance": "standard dog appearance"
            }
            
        except Exception as e:
            logging.error(f"Error extracting visual features: {str(e)}")
            return {
                "colors": "unknown colors",
                "appearance": "standard dog appearance"
            }
    
    def analyze_visual_details(self, image):
        """Extract visual details from the image using BLIP model."""
        try:
            visual_features = self._extract_visual_features(image)
            
            # Initialize with direct observations as fallback
            details = {
                "caption": "A dog in an image",
                "colors": visual_features.get("colors", "unknown"),
                "appearance": visual_features.get("appearance", "unknown")
            }
            
          
            caption_prompts = [
                "Describe this dog in one sentence.",
                "What breed of dog is in this image?",
                "Describe what you see in this image.",
                "Caption this image of a dog."
            ]
            
            captions = []
            if self.blip_model is not None:
                for prompt in caption_prompts:
                    try:
                        caption = self._process_visual_query(image, prompt)
                        if caption and caption.lower() not in ["i don't know", "unknown", "i do not know", "i cannot tell"]:
                            captions.append(caption)
                    except Exception as e:
                        logging.error(f"Error getting caption with prompt '{prompt}': {str(e)}")

                if captions:
                    # Use the most informative caption (usually the longest one that's not too long)
                    captions.sort(key=lambda x: len(x), reverse=True)
                   
                    valid_captions = [c for c in captions if len(c) < 150]
                    if valid_captions:
                        details["caption"] = valid_captions[0]
                    elif captions:
                        details["caption"] = captions[0]

            
            color_prompts = [
                "What color is this dog's coat?",
                "Describe the color of this dog.",
                "What are the coat colors of this dog in the image?",
                "List all colors visible on this dog."
            ]
            
            colors = []
            if self.blip_model is not None:
                for prompt in color_prompts:
                    try:
                        color = self._process_visual_query(image, prompt)
                        if color and color.lower() not in ["i don't know", "unknown", "i do not know", "i cannot tell"]:
                            colors.append(color)
                    except Exception as e:
                        logging.error(f"Error getting color with prompt '{prompt}': {str(e)}")

                if colors:
                  
                    colors.sort(key=lambda x: len(x), reverse=True)
                    valid_colors = [c for c in colors if len(c) < 100]
                    if valid_colors:
                        details["colors"] = valid_colors[0]
                    elif colors:
                        details["colors"] = colors[0]
            
            # Try to get better appearance details
            appearance_prompts = [
                "Describe the physical appearance of this dog.",
                "What distinctive features does this dog have?",
                "How would you describe this dog's physical traits?",
                "Describe the dog's body, legs, and face in this image."
            ]
            
            appearances = []
            if self.blip_model is not None:
                for prompt in appearance_prompts:
                    try:
                        appearance = self._process_visual_query(image, prompt)
                        if appearance and appearance.lower() not in ["i don't know", "unknown", "i do not know", "i cannot tell"]:
                            appearances.append(appearance)
                    except Exception as e:
                        logging.error(f"Error getting appearance with prompt '{prompt}': {str(e)}")

            if appearances:
                # Use the most informative appearance description
                appearances.sort(key=lambda x: len(x), reverse=True)
                valid_appearances = [a for a in appearances if len(a) < 200]
                if valid_appearances:
                    details["appearance"] = valid_appearances[0]
                elif appearances:
                    details["appearance"] = appearances[0]

            return details
            
        except Exception as e:
            logging.error(f"Error in visual analysis: {str(e)}")
            return {
                "caption": "Error analyzing image",
                "colors": "Unknown",
                "appearance": "Could not determine"
            }

    def _process_visual_query(self, image, query):
        """Process a single visual query with proper error handling.
        
        Args:
            image: The image to analyze
            query: The specific query to process
            
        Returns:
            The response text or None if processing failed
        """
        try:
        
            try:
                from deployment_config import get_deployment_settings
                settings = get_deployment_settings()
            except ImportError:
                settings = {
                    "blip_max_length": 100
                }
                
          
            if not self.blip_initialized:
                if not self._load_blip_model():
                    return None
                    
         
            inputs = self.blip_processor(image, query, return_tensors="pt")
            
           
            import signal
            
            class TimeoutException(Exception):
                pass
            
            def timeout_handler(signum, frame):
                raise TimeoutException("BLIP inference timed out")
            
           
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10) 
            
            try:
              
                with torch.no_grad():
                    outputs = self.blip_model.generate(
                        **inputs, 
                        max_length=settings.get("blip_max_length", 100)
                    )
                answer = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                
               
                signal.alarm(0)
                
               
                if answer and answer.strip() not in ["", ".", "?", "unknown", "not sure"]:
                    return answer
                return None
                
            except TimeoutException:
                logger.warning(f"BLIP inference timed out for query: {query}")
                return None
            finally:
               
                signal.alarm(0)
                
        except Exception as e:
            logger.error(f"Error processing visual query '{query}': {str(e)}")
            return None

def main():
    """Test function to demonstrate VLM functionality."""
    from PIL import Image
    import os
    
    vlm = DogBreedVLM()
    
 
    sample_path = os.path.join('datasets', 'train', next(os.listdir(os.path.join('datasets', 'train'))))
    sample_image = Image.open(sample_path).convert('RGB')
    
   
    predictions = vlm.process_image(sample_image)
    
    print(f"Detected breed: {predictions[0]['breed']} ({predictions[0]['confidence']:.2%})")
    print(f"Description: {predictions[0]['description']}")
    print(f"Visual reasoning: {predictions[0]['visual_reasoning']}")
    print(f"Confidence: {predictions[0]['confidence_statement']}")
    
   
    visual_comparison = vlm.visual_reasoning(predictions)
    print(f"\nVisual comparison: {visual_comparison}")
    
   
    print("\nVisual Analysis:")
    visual_details = vlm.analyze_visual_details(sample_image)
    for key, value in visual_details.items():
        print(f"{key.title()}: {value}")
    
    
    sample_queries = [
        "How big is this dog breed?",
        "What color is this dog?",
        "Is this dog good with children?",
        "What's the energy level of this breed?",
        "How does this breed compare to similar breeds?",
        "Where does this breed come from?",
        "Why do you think this is this particular breed?",
        "How confident are you about this identification?"
    ]
    
    print("\nQuery responses:")
    for query in sample_queries:
        response = vlm.answer_query(query, predictions)
        print(f"Q: {query}")
        print(f"A: {response}\n")

if __name__ == "__main__":
    main() 