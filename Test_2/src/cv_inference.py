import torch
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Настройки
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:\Users\Greta\Documents\Учеба\PythonProject\Winstars\Test2\models\image_model.pth"

# -----------------------------
# Модель ResNet18
# -----------------------------
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # количество классов

# Загружаем веса из чекпойнта
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# -----------------------------
# Преобразования изображений
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Классы животных
CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# -----------------------------
# Функция классификации изображения
# -----------------------------
def classify_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)

    return CLASSES[pred.item()]
