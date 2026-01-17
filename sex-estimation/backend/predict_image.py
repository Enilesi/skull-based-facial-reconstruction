import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sex_cnn.pt"
IMG_PATH = "skull-female.jpg"

IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 1)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

img = Image.open(IMG_PATH).convert("RGB")
x = tf(img).unsqueeze(0).to(device)

with torch.no_grad():
    prob_male = torch.sigmoid(model(x)).item()

sex = "M" if prob_male >= 0.5 else "F"

print("Sex:", sex)
print("Male probability:", prob_male)
