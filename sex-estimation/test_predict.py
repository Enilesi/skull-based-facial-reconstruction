import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

base = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base, "models", "sex_effnet.pt")
IMAGE_PATH = os.path.join(base, "image.png")

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
x = tf(img).unsqueeze(0).to(device)

with torch.no_grad():
    y = torch.softmax(model(x), 1)[0]

print("female" if y[0] > y[1] else "male")
print(float(y[0]), float(y[1]))
