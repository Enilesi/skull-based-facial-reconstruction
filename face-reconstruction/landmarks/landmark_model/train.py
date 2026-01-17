import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import LandmarkCNN
from dataset import SkullLandmarkDataset

import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
LANDMARKS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(LANDMARKS_DIR)

from landmark_slots import NUM_LANDMARK_SLOTS

BATCH_SIZE = 2
EPOCHS = 200
LR = 1e-3

LANDMARK_ROOT = "../../landmarked"
MODEL_PATH = "landmark_model.pth"

train_ds = SkullLandmarkDataset(LANDMARK_ROOT, split="train")
val_ds   = SkullLandmarkDataset(LANDMARK_ROOT, split="val")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = LandmarkCNN(NUM_LANDMARK_SLOTS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, coords in train_loader:
        preds = model(imgs)
        loss = criterion(preds, coords)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, coords in val_loader:
            preds = model(imgs)
            loss = criterion(preds, coords)
            val_loss += loss.item()

    val_loss /= max(1, len(val_loader))

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
