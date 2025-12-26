import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
from pathlib import Path


class BirdCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(256 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



TRAIN_DIR = Path("data/prepared/train")
CLASS_NAMES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BirdCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("models/bird_cnn.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict(image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    result = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите изображение птицы"),
    outputs=gr.Label(num_top_classes=3, label="Результат классификации"),
    title="Классификация видов птиц",
    description="CNN-модель для определения вида птицы по изображению"
)

if __name__ == "__main__":
    demo.launch()
