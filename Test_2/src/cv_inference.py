import os
import torch
from torchvision import models, transforms
from PIL import Image


# Settings / Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Relative path to model to make it cross-platform
MODEL_PATH = os.path.join("Test_2", "models", "image_model.pth")


# ResNet18 Model
# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer to match the number of classes (10)
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Load weights from checkpoint only if the file exists
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode
else:
    print(f"WARNING: Model checkpoint not found at '{MODEL_PATH}'. CV predictions will be random.")
    model = None  # Set model to None to avoid using it


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor()  # Convert image to PyTorch tensor
])

# Animal classes
CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']



# Image classification function
def classify_image(image_path: str) -> str:
    #Classify an image using the pre-trained model.

    if model is None:
        return "unknown"

    # Open the image and ensure it's in RGB mode
    img = Image.open(image_path).convert("RGB")

    # Apply transformations and add batch dimension
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    # Forward pass without computing gradients
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)  # Get predicted class index

    # Return class label
    return CLASSES[pred.item()]
