import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import os

DATA_DIR = "/data/images"
MODEL_OUT = "sex_effnet.pt"

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
FOLDS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
targets = np.array([y for _, y in full_ds])

print("Classes:", full_ds.classes)
print("Total images:", len(full_ds))


def get_model():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model.to(device)


skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"\n=== Fold {fold+1}/{FOLDS} ===")

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_tf), val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        best = max(best, acc)
        print(f"Epoch {epoch+1}/{EPOCHS} | Val Acc: {acc:.3f}")

    fold_accuracies.append(best)


print("\n=== FINAL RESULTS ===")
print("Fold accuracies:", fold_accuracies)
print("Mean accuracy:", np.mean(fold_accuracies))

torch.save(model.state_dict(), MODEL_OUT)
print("Saved model to:", MODEL_OUT)
