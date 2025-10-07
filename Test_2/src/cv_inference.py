import os
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the model checkpoint


# Path relative to this file
MODEL_PATH = Path(__file__).parent.parent / "models" / "image_model.pth"

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at '{MODEL_PATH}'")

# Load the checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Create a ResNet18 model and adjust the final fully connected layer
num_classes = len(checkpoint['class_names'])
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()  # Set model to evaluation mode

# Load class names from the checkpoint
CLASSES = checkpoint['class_names']

# Define image transformations (must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the same size as during training
    transforms.ToTensor(),           # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                         [0.229, 0.224, 0.225])  # and standard deviation
])

def classify_image(image_path: str) -> str:
    """
    Classifies an image and returns the predicted class name.
    Always returns a valid class from CLASSES.
    """
    # Open the image and convert to RGB
    img = Image.open(image_path).convert("RGB")

    # Apply transformations and add batch dimension
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    # Perform inference without computing gradients
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)  # Get the index of the max probability

    # Return the class name corresponding to the predicted index
    return CLASSES[pred.item()]
