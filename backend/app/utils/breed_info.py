def get_all_breeds():
    """Returns a list of all supported dog breeds."""
    return [
        "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale", "american_staffordshire_terrier",
        "appenzeller", "australian_terrier", "basenji", "basset", "beagle", "bedlington_terrier", "bernese_mountain_dog",
        "black-and-tan_coonhound", "blenheim_spaniel", "bloodhound", "bluetick", "border_collie", "border_terrier",
        "borzoi", "boston_bull", "bouvier_des_flandres", "boxer", "brabancon_griffon", "brittany_spaniel",
        "bull_mastiff", "cairn", "cardigan", "chesapeake_bay_retriever", "chihuahua", "chow", "clumber",
        "cocker_spaniel", "collie", "curly-coated_retriever", "dandie_dinmont", "dhole", "dingo", "doberman",
        "english_foxhound", "english_setter", "english_springer", "entlebucher", "eskimo_dog", "flat-coated_retriever",
        "french_bulldog", "german_shepherd", "giant_schnauzer", "golden_retriever", "gordon_setter", "great_dane",
        "great_pyrenees", "greater_swiss_mountain_dog", "groenendael", "ibizan_hound", "irish_setter", "irish_terrier",
        "irish_water_spaniel", "irish_wolfhound", "italian_greyhound", "japanese_spaniel", "keeshond", "kelpie",
        "kerry_blue_terrier", "komondor", "kuvasz", "labrador_retriever", "lakeland_terrier", "leonberg", "lhasa",
        "malamute", "malinois", "maltese_dog", "mexican_hairless", "miniature_pinscher", "miniature_poodle",
        "miniature_schnauzer", "newfoundland", "norfolk_terrier", "norwegian_elkhound", "norwich_terrier",
        "old_english_sheepdog", "otterhound", "papillon", "pekinese", "pembroke", "pomeranian", "pug", "redbone",
        "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki", "samoyed", "schipperke", "scotch_terrier",
        "scottish_deerhound", "sealyham_terrier", "shetland_sheepdog", "shih-tzu", "siberian_husky", "silky_terrier",
        "soft-coated_wheaten_terrier", "staffordshire_bullterrier", "standard_poodle", "standard_schnauzer",
        "sussex_spaniel", "tibetan_mastiff", "tibetan_terrier", "toy_poodle", "toy_terrier", "vizsla", "walker_hound",
        "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", "whippet", "wire-haired_fox_terrier",
        "yorkshire_terrier"
    ]

# Dictionary with detailed breed information including appearances and origins
DETAILED_BREED_INFO = {
    "golden_retriever": {
        "description": "An intelligent, friendly, and devoted dog. Excels at retrieving game for hunters, tracking, and finding people.",
        "characteristics": ["Intelligent", "Friendly", "Reliable", "Trustworthy", "Confident"],
        "size": "Large",
        "energy_level": "High",
        "good_with_children": True,
        "appearance": "thick golden coat, friendly expression, sturdy build",
        "origin": "Scotland",
        "purpose": "retrieving game for hunters",
        "lifespan": "10-12 years",
        "group": "Sporting"
    },
    "german_shepherd": {
        "description": "A versatile working dog, capable of performing many tasks, including guiding the blind and law enforcement.",
        "characteristics": ["Intelligent", "Loyal", "Confident", "Courageous", "Steady"],
        "size": "Large",
        "energy_level": "High",
        "good_with_children": True,
        "appearance": "pointed ears, strong jaw, wolf-like face, tan and black coloring",
        "origin": "Germany",
        "purpose": "herding and police/military work",
        "lifespan": "9-13 years",
        "group": "Herding"
    },
    "beagle": {
        "description": "A small scent hound developed primarily for hunting hare. Known for its great sense of smell.",
        "characteristics": ["Merry", "Friendly", "Curious", "Clever", "Energetic"],
        "size": "Small to Medium",
        "energy_level": "High",
        "good_with_children": True,
        "appearance": "tricolor coat (black, white and tan), floppy ears, compact body",
        "origin": "England",
        "purpose": "hunting small game, especially rabbits",
        "lifespan": "12-15 years",
        "group": "Hound"
    },
    "pug": {
        "description": "A small companion dog with a wrinkled face and curled tail. Known for being charming and mischievous.",
        "characteristics": ["Charming", "Mischievous", "Loving", "Stubborn", "Sociable"],
        "size": "Small",
        "energy_level": "Low",
        "good_with_children": True,
        "appearance": "wrinkled face, short muzzle, curled tail, fawn or black coat",
        "origin": "China",
        "purpose": "companion dog for nobility",
        "lifespan": "12-15 years",
        "group": "Toy"
    },
    "samoyed": {
        "description": "A friendly and alert herding dog with a thick white coat that helped them survive in cold Siberian climates.",
        "characteristics": ["Friendly", "Alert", "Adaptable", "Intelligent", "Gentle"],
        "size": "Medium to Large",
        "energy_level": "High",
        "good_with_children": True,
        "appearance": "thick white fluffy coat, smiling expression, erect triangular ears",
        "origin": "Siberia",
        "purpose": "herding reindeer and pulling sleds",
        "lifespan": "12-14 years",
        "group": "Working"
    },
    "bernese_mountain_dog": {
        "description": "A large working dog from the Swiss Alps, known for its calm temperament and distinctive tri-color coat.",
        "characteristics": ["Good-natured", "Calm", "Strong", "Affectionate", "Loyal"],
        "size": "Large",
        "energy_level": "Medium",
        "good_with_children": True,
        "appearance": "tri-color coat (black, white, and rust), strong build, gentle expression",
        "origin": "Switzerland",
        "purpose": "farm work, pulling carts, and guarding",
        "lifespan": "7-10 years",
        "group": "Working"
    }
}

def get_breed_info(breed):
    """Returns detailed information about a specific breed."""
    # First check if we have detailed info for this breed
    if breed in DETAILED_BREED_INFO:
        return DETAILED_BREED_INFO[breed]
    
    # Check if we have basic info in the simple dictionary
    if breed in BREED_INFO:
        return BREED_INFO[breed]
    
    # Otherwise, return a generic template with the breed name
    breed_name = breed.replace('_', ' ').title()
    
    # Generate size based on breed name (just an example - in a real app, you'd use actual data)
    if any(word in breed.lower() for word in ['toy', 'miniature', 'small', 'chihuahua', 'terrier', 'spaniel']):
        size = "Small"
    elif any(word in breed.lower() for word in ['mastiff', 'shepherd', 'retriever', 'hound']):
        size = "Large"
    else:
        size = "Medium"
    
    # Generate energy level based on breed name
    if any(word in breed.lower() for word in ['hound', 'terrier', 'collie', 'shepherd', 'retriever']):
        energy = "High"
    elif any(word in breed.lower() for word in ['mastiff', 'bulldog', 'shih-tzu']):
        energy = "Low"
    else:
        energy = "Medium"
        
    return {
        "name": breed_name,
        "description": f"The {breed_name} is a distinctive breed with unique characteristics and temperament.",
        "characteristics": ["Loyal", "Intelligent", "Adaptable"],
        "size": size,
        "energy_level": energy,
        "good_with_children": True
    }

# Dictionary with basic breed information
BREED_INFO = {
    "affenpinscher": {
        "description": "A small but fearless dog with a monkey-like expression. Originally bred to hunt rodents.",
        "characteristics": ["Alert", "Inquisitive", "Adventurous", "Stubborn", "Playful", "Confident"],
        "size": "Small",
        "energy_level": "Moderate",
        "good_with_children": True
    },
    "bulldog": {
        "description": "A muscular, heavy dog with a wrinkled face and a distinctive pushed-in nose.",
        "characteristics": ["Docile", "Willful", "Friendly", "Gregarious"],
        "size": "Medium",
        "energy_level": "Low",
        "good_with_children": True
    },
    "shetland_sheepdog": {
        "description": "A small, active, and agile herding dog, highly intelligent and eager to please.",
        "characteristics": ["Intelligent", "Energetic", "Loyal", "Vocal", "Alert"],
        "size": "Small to Medium",
        "energy_level": "High", 
        "good_with_children": True,
        "appearance": "long, thick coat, usually sable, with white markings and a ruff around the neck"
    },
    "scottish_deerhound": {
        "description": "A tall, dignified sighthound bred to hunt red deer by coursing.",
        "characteristics": ["Dignified", "Gentle", "Polite", "Independent", "Athletic"],
        "size": "Large",
        "energy_level": "Medium",
        "good_with_children": True,
        "appearance": "rough coat, long legs, deep chest, and a curved tail"
    }
} 