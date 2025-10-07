import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms

# -----------------------------
# 1️⃣ NER: Simple dictionary lookup
# -----------------------------
ANIMALS = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def extract_animals(text: str):
    """
    Extract animal names from text using a simple dictionary lookup.
    """
    text_lower = text.lower()
    found_animals = [animal for animal in ANIMALS if animal in text_lower]
    return found_animals

# -----------------------------
# 2️⃣ CV: ResNet18 inference
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).parent.parent / "models" / "image_model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Load ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_classes = len(checkpoint['class_names'])
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

CLASSES = checkpoint['class_names']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_image(image_path: Path) -> str:
    """
    Classify an image and return predicted animal class.
    """
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    return CLASSES[pred.item()]

# -----------------------------
# 3️⃣ Pipeline function
# -----------------------------
DATA_DIR = Path(__file__).parent.parent / "data" / "images"
classes = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]

def animal_pipeline(text: str, image_path: Path = None) -> bool:
    """
    Full pipeline: NER + CV.
    If image_path is None, choose a random image from dataset.
    Returns True if text matches image, False otherwise.
    """
    # Extract animals from text
    animals_from_text = extract_animals(text)
    if not animals_from_text:
        print("No animals found in text.")
        return False

    # Choose random image if none provided
    if image_path is None:
        image_class = random.choice(classes)
        image_path = random.choice(list((DATA_DIR / image_class).iterdir()))

    # Classify image
    predicted_animal = classify_image(image_path)

    # Compare text vs image
    match = any(predicted_animal.lower() in a.lower() for a in animals_from_text)

    # Show image with result
    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Text: '{text}'\nPredicted: {predicted_animal} | Match: {match}")
    plt.show(block=True)

    return match

# -----------------------------
# 4️⃣ Example test
# -----------------------------
if __name__ == "__main__":
    for _ in range(5):
        text_class = random.choice(classes)
        text = f"There is a {text_class} in the picture"
        animal_pipeline(text)  # image is random
