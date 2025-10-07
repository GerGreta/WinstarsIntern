# List of animals corresponding to classes in the CV model
ANIMALS = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def extract_animals(text: str):
    #    Extracts animal names from a given text using a simple dictionary lookup.
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Find all animals that appear in the text
    found_animals = [animal for animal in ANIMALS if animal in text_lower]

    return found_animals
