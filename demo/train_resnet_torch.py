import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# --- SETTINGS ---
base_dir = "."
classes = ["female skulls", "male skulls"]
img_size = (224, 224)
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- AUGMENTATIONS & NORMALIZATION ---
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- CUSTOM DATASET ---
class SkullDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# --- LOAD FILES ---
image_paths, labels = [], []
for label, folder in enumerate(classes):
    folder_path = os.path.join(base_dir, folder)
    for img_name in os.listdir(folder_path):
        image_paths.append(os.path.join(folder_path, img_name))
        labels.append(label)

# --- SPLIT ---
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

train_ds = SkullDataset(train_paths, train_labels, transform=train_transform)
test_ds = SkullDataset(test_paths, test_labels, transform=test_transform)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# --- MODEL (TRANSFER LEARNING) ---
model = models.resnet18(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# --- TRAIN ---
for epoch in range(10):
    model.train()
    total_loss = 0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}")

# --- EVALUATE ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for Xb, yb in test_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        preds = model(Xb)
        predicted = torch.argmax(preds, dim=1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

print(f"\nTest accuracy: {correct/total:.2f}")
torch.save(model.state_dict(), "skull_resnet18.pt")
print("Model saved as skull_resnet18.pt")
