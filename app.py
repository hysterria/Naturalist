import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import gradio as gr

# -----------------------------
# Пути к модели
# -----------------------------
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "bird_cnn.pth"

# -----------------------------
# Список классов
# -----------------------------
CLASSES = [
    "Белоспинный дятел",
    "Ворон",
    "Клёст-еловик",
    "Снегирь",
    "Уральская неясыть",
    "Чёрный коршун"
]

# -----------------------------
# Определяем модель (как у тебя)
# -----------------------------
class BirdCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(256*8*8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

# -----------------------------
# Загружаем модель
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdCNN(len(CLASSES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# Преобразования изображений
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# -----------------------------
# Функция предсказания
# -----------------------------
def predict(image):
    """
    image: PIL.Image, загруженное пользователем
    """
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    class_name = CLASSES[pred.item()]
    return class_name

# -----------------------------
# Создаем интерфейс Gradio
# -----------------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Predicted Class"),
    title="Bird Classifier",
    description="Загрузите изображение птицы и модель предскажет вид."
)

# -----------------------------
# Запуск приложения
# -----------------------------
if __name__ == "__main__":
    iface.launch()
