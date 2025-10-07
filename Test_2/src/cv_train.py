import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main():

    # Paths and parameters
    ROOT = Path(__file__).resolve().parent.parent  # project root
    DATA_DIR = ROOT / "data" / "images"
    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    batch_size = 32
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # resize for faster training
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    loader = DataLoader(
        datasets.ImageFolder(DATA_DIR, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    dataset = loader.dataset
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Model setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss={total_loss/len(loader):.4f}, Accuracy={correct/len(dataset):.4f}")

    # Save trained model
    model_path = MODELS_DIR / "image_model.pth"
    torch.save({'model_state_dict': model.state_dict(),
                'class_names': class_names}, model_path)
    print(f"✅ CV model saved to: {model_path}")

if __name__ == "__main__":
    main()