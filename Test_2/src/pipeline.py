import os
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add src folder to sys.path so we can import custom modules
import sys
ROOT = Path(__file__).parent.parent  # project root
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# Import custom functions
from ner_inference import extract_animals
from cv_inference import classify_image

# Path to the images folder
DATA_DIR = ROOT / "data" / "images"
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

# -----------------------------
# Pipeline function
# -----------------------------
def pipeline(text: str = None, show_image: bool = True) -> bool:
    """
    Process a text description and a random image, compare them, and return True if match.

    Args:
        text (str): User-provided text describing the animal. If None, input() will be used.
        show_image (bool): Whether to display the image using matplotlib.

    Returns:
        bool: True if the animal in the image matches the text, False otherwise.
    """
    # 1️⃣ Get text input from user if not provided
    if text is None:
        text = input("Enter text describing the animal: ")

    # 2️⃣ Select a random image
    image_class = random.choice(classes)
    folder_path = os.path.join(DATA_DIR, image_class)
    img_file = random.choice(os.listdir(folder_path))
    img_path = os.path.join(folder_path, img_file)

    # 3️⃣ Display the image
    if show_image:
        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Random image from folder: {image_class}")
        plt.show()

    # 4️⃣ Extract animals from text (NER)
    animals_extracted = extract_animals(text)
    print("Text:", text)
    print("Extracted from text:", animals_extracted)

    # 5️⃣ Classify the animal in the image (CV)
    animal_from_image = classify_image(img_path)
    print("Predicted from image:", animal_from_image)

    # 6️⃣ Compare text vs image
    match = any(animal_from_image.lower() in a.lower() for a in animals_extracted)
    print("Match:", match)
    print("-" * 50)

    return match

# -----------------------------
# Demo run
# -----------------------------
if __name__ == "__main__":
    num_examples = 3  # number of demo runs
    for i in range(num_examples):
        print(f"Demo example {i+1}/{num_examples}")
        pipeline()
