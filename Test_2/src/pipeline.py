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

# Path to the images folder (relative)
DATA_DIR = ROOT / "data" / "images"

def pipeline(text: str, image_path: Path) -> bool:
    # Analyze text
    print("🧠 Text analysis...")
    animals_from_text = extract_animals(text)
    print(f"Extracted from text: {animals_from_text}")

    # Show image
    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image: {image_path.name}")
    plt.show(block=True)  # ensure window opens in PyCharm

    # Classify image
    animal_from_image = classify_image(image_path)
    print(f"Identified in image: {animal_from_image}")

    # Compare text vs image
    result = any(animal_from_image.lower() in a.lower() for a in animals_from_text)
    print(f"✅ Match: {result}\n")
    return result

# Iterate over all classes and pick one random image per class
for class_name in os.listdir(DATA_DIR):
    folder_path = DATA_DIR / class_name
    if not folder_path.is_dir():
        continue  # skip non-folders

    img_files = [f for f in folder_path.iterdir() if f.is_file()]
    if not img_files:
        continue

    img_path = random.choice(img_files)
    text = f"There is a {class_name} in the picture"
    pipeline(text, img_path)