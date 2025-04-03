import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DogGeneralQA:
    """A lightweight question-answering model for general dog questions."""
    
    def __init__(self):
        """Initialize the QA model with a small pretrained model."""
        self.model = None
        self.tokenizer = None
        self.context = self._load_dog_context()
        self.is_initialized = False
        
    def initialize(self):
        """Lazy initialization to load model only when needed"""
        try:
            # Use a small QA model - distilbert is much smaller than BERT
            model_name = "distilbert-base-cased-distilled-squad"
            
            logger.info(f"Loading QA model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Put in evaluation mode
            self.model.eval()
            
            self.is_initialized = True
            logger.info("QA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading QA model: {str(e)}")
            raise
    
    def _load_dog_context(self):
        """Load general knowledge about dogs."""
        return """
        Dogs are domesticated mammals, part of the wolf family. They were originally bred from wolves.
        They have been bred by humans for a long time, and were the first animals ever to be domesticated.
        
        Dogs are commonly known as man's best friend. They are loyal, friendly, and protective animals that have lived alongside humans for thousands of years.
        
        Different dog breeds have different temperaments, which is most clearly visible in different breeds of working dogs.
        
        Dogs can have many different jobs. Some dogs are pets, some work for the police to find criminals or dangerous items like bombs or drugs, some help people who cannot see or have other health problems.
        
        Dogs need to be taken care of by humans. They need a balanced diet, regular exercise, grooming, training, veterinary care, and a lot of love and attention.
        
        Puppies are baby dogs. They grow to adult size in about 12 to 24 months, depending on the breed.
        
        Dogs communicate through body language, vocalizations like barking, growling, or whining, and scent marking. They can understand many human words and commands.
        
        Dogs have a highly developed sense of smell, much stronger than humans. They also have excellent hearing but their color vision is not as good as humans.
        
        Common dog health issues include fleas and ticks, ear infections, worms, dental problems, obesity, and allergies. Regular vet check-ups help catch issues early.
        
        Dogs should be trained using positive reinforcement methods. Basic commands like sit, stay, come, and heel help keep dogs safe and well-behaved.
        
        Dog breeds are categorized into groups based on their original purpose: herding dogs, working dogs, terriers, sporting dogs, non-sporting dogs, hounds, and toy breeds.
        
        Small dog breeds include Chihuahua, Pomeranian, Shih Tzu, and Yorkshire Terrier. Medium dog breeds include Beagle, Bulldog, and Cocker Spaniel. Large dog breeds include German Shepherd, Golden Retriever, and Labrador Retriever. Giant dog breeds include Great Dane, Saint Bernard, and Newfoundland.
        
        Dogs should be socialized early to help them become comfortable around different people, animals, and environments.
        
        The average lifespan of a dog depends on the breed, but it's typically between 10 and 13 years. Small breeds generally live longer than large breeds.
        """
    
    def expand_context(self, breed_info):
        """Add breed-specific information to the context."""
        breed_name = breed_info.get("breed", "").replace("_", " ").capitalize()
        
        breed_context = f"""
        The {breed_name} is a {breed_info.get('size', 'medium-sized')} dog breed. 
        This breed is known for being {', '.join(breed_info.get('characteristics', ['friendly']))}. 
        They have {breed_info.get('energy_level', 'moderate')} energy levels and are {breed_info.get('good_with_children', 'generally good')} with children.
        """
        
        return self.context + breed_context
    
    def answer_question(self, question, top_breed_info=None):
        """Answer a general question about dogs using the QA model."""
        if not self.is_initialized:
            self.initialize()
            
        if not self.is_initialized:
            return "I'm sorry, I couldn't load my knowledge base to answer that question."
            
        try:
            # Use expanded context if breed info provided
            context = self.expand_context(top_breed_info) if top_breed_info else self.context
            
            # Encode the question and context
            inputs = self.tokenizer.encode_plus(
                question, 
                context, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Get the model's prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get the most likely answer span
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            
            # Convert token IDs to tokens and join them for the answer
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][0][answer_start:answer_end]
                )
            )
            
            # If answer is empty or just punctuation, give a fallback
            if not answer or answer.strip() in [".", ",", "?", "!", ""]:
                return "I don't have specific information to answer that question about dogs."
                
            return answer
            
        except Exception as e:
            logger.error(f"Error in QA model: {str(e)}")
            return "I'm having trouble processing that question. Please try a different question."
            
    def is_general_dog_question(self, question):
        """Check if this is a general dog question vs. a breed-specific question."""
        breed_specific_terms = [
            'this breed', 'this dog breed', 'breed in the image',
            'shown breed', 'pictured breed', 'in the picture'
        ]
        
        # Check if this is likely a general question
        question = question.lower()
        for term in breed_specific_terms:
            if term in question:
                return False
                
        # Check for general dog question patterns
        general_patterns = [
            'all dogs', 'dogs usually', 'most dogs', 'dogs generally',
            'can dogs', 'do dogs', 'why do dogs', 'how do dogs'
        ]
        
        for pattern in general_patterns:
            if pattern in question:
                return True
                
        # Default to false if no strong indicators
        return False

# Singleton instance
general_qa = DogGeneralQA()

def get_qa_model():
    """Get the singleton QA model instance."""
    return general_qa 