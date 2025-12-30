from dataset import SkullLandmarkDataset
from model import LandmarkCNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

LANDMARK_ROOT = "../landmarked"
BATCH_SIZE = 4
EPOCHS = 200
LR = 1e-3

dataset = SkullLandmarkDataset(LANDMARK_ROOT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_landmarks = len(dataset[0][1]) // 2

model = LandmarkCNN(num_landmarks)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    for imgs, coords in loader:
        preds = model(imgs)
        loss = criterion(preds, coords)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "landmark_model.pth")
print("Model saved to landmark_model.pth")
