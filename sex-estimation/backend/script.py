import os
import shutil
from pathlib import Path
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# ======================
# CONFIG
# ======================

ROOT = Path(__file__).resolve().parents[0]
DATA_SRC = ROOT / "data" / "images"
DATA_SPLIT = ROOT / "data_split"
MODEL_OUT = ROOT / "models" / "sex_effnet.pt"

IMG_SIZE = 224
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ======================
# 1. CLEAN + SPLIT DATA
# ======================

def clean_split_folders():
    if DATA_SPLIT.exists():
        shutil.rmtree(DATA_SPLIT)
    for folder in ["train", "val", "test"]:
        for cls in ["male", "female"]:
            (DATA_SPLIT / folder / cls).mkdir(parents=True, exist_ok=True)

def split_data(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    print("\n=== Splitting dataset ===")
    clean_split_folders()

    for cls in ["male", "female"]:
        src_folder = DATA_SRC / cls
        images = list(src_folder.glob("*.*"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]

        for img in train_imgs:
            shutil.copy(img, DATA_SPLIT / "train" / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, DATA_SPLIT / "val" / cls / img.name)
        for img in test_imgs:
            shutil.copy(img, DATA_SPLIT / "test" / cls / img.name)

        print(f"{cls}: {n_total} images → {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")

# DO SPLIT
split_data()

# ======================
# 2. TRAINING
# ======================

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.15, 0.15),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_datasets():
    train_ds = datasets.ImageFolder(DATA_SPLIT / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_SPLIT / "val", transform=val_tf)
    test_ds = datasets.ImageFolder(DATA_SPLIT / "test", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=1)

    print("\nLoaded:")
    print(" Train:", len(train_ds))
    print(" Val:", len(val_ds))
    print(" Test:", len(test_ds))

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = load_datasets()

# Compute class weights
class_counts = [0, 0]
for _, y in datasets.ImageFolder(DATA_SRC):
    class_counts[y] += 1

weights = torch.tensor([1/class_counts[0], 1/class_counts[1]]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Load EfficientNet
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

for name, param in model.named_parameters():
    if "features.6" in name or "features.7" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ======================
# TRAIN LOOP
# ======================

best_acc = 0.0

print("\n=== Training model ===")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = torch.softmax(model(x), dim=1).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss:.3f} | Val Acc {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_OUT)
        print("✔ Saved improved model")

print("\nTraining complete.")
print("Best model saved to:", MODEL_OUT)

# ======================
# 3. TEST ON ALL TEST IMAGES
# ======================

print("\n=== Testing on all test images ===")

model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
model.eval()

for img_path in sorted((DATA_SPLIT / "test" / "male").glob("*.*")) + \
                 sorted((DATA_SPLIT / "test" / "female").glob("*.*")):

    img = Image.open(img_path).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        male = probs[1].item()
        female = probs[0].item()

    pred = "male" if male > female else "female"
    print(f"{img_path.name:25s} → {pred.upper()}   (M={male:.3f}, F={female:.3f})")
