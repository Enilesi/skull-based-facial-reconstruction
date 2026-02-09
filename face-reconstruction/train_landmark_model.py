import os
import sys
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from landmarks.landmark_slots import LANDMARK_SLOTS, NUM_LANDMARK_SLOTS
from landmarks.landmarks_schema import LANDMARKS as LANDMARKS_SCHEMA, SKIP_LANDMARKS

LANDMARK_COLORS = {
    1:(0,0,255),2:(0,165,255),3:(0,255,255),4:(0,255,0),5:(255,128,0),
    6:(128,0,128),7:(255,0,255),8:(0,0,128),9:(114,128,250),10:(208,224,64),
    11:(225,105,65),12:(0,128,0),13:(50,205,50),14:(128,128,0),15:(0,215,255),
    16:(130,0,75),18:(19,69,139),19:(203,192,255),20:(128,0,0),21:(0,128,128)
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

SEX_FOLDERS = [
    ("male", "landmarked_male"),
    ("female", "landmarked_female"),
    ("unasigned", "landmarked_unasigned"),
    ("unassigned", "landmarked_unasigned"),
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json_map():
    root = ROOT / "data" / "landmarked"
    mapping = {}
    for sex, folder in SEX_FOLDERS:
        p = root / folder
        if not p.exists():
            continue
        for j in p.glob("*_landmarks.json"):
            img_id = j.stem.replace("_landmarks", "")
            mapping[(sex, img_id)] = str(j)
    return mapping

def collect_images():
    root = ROOT / "data" / "images"
    images = []
    for sex, _ in SEX_FOLDERS:
        p = root / sex
        if not p.exists():
            continue
        for fp in p.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
                images.append((sex, str(fp)))
    return images

def split_train_val_test(seed=42, train_ratio=0.7, val_ratio=0.15):
    set_seed(seed)
    json_map = load_json_map()
    images = collect_images()

    paired = []
    test = []

    for sex, img_path in images:
        img_id = Path(img_path).stem
        jp = json_map.get((sex, img_id))
        if jp is None:
            test.append((sex, img_path))
        else:
            paired.append((sex, img_path, jp))

    random.shuffle(paired)
    n = len(paired)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train = paired[:n_train]
    val = paired[n_train:n_train+n_val]
    return train, val, test

def parse_json_to_slots(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    grouped = {}
    for lm in data.get("landmarks", []):
        lid = int(lm.get("id", -1))
        if lid in SKIP_LANDMARKS:
            continue
        if lid not in LANDMARKS_SCHEMA:
            continue
        grouped.setdefault(lid, []).append((float(lm["x"]), float(lm["y"])))

    slot_points = []
    for slot in LANDMARK_SLOTS:
        lid = int(slot["id"])
        side = slot["side"]
        name, _, count = LANDMARKS_SCHEMA[lid]

        pts = grouped.get(lid, [])
        if count == 1:
            if len(pts) < 1:
                return None
            slot_points.append(pts[0])
        else:
            if len(pts) < 2:
                return None
            pts2 = sorted(pts[:2], key=lambda p: p[0])
            if side == "left":
                slot_points.append(pts2[0])
            else:
                slot_points.append(pts2[1])

    if len(slot_points) != NUM_LANDMARK_SLOTS:
        return None

    return np.array(slot_points, dtype=np.float32)

def make_heatmaps(points_xy, out_h, out_w, sigma):
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)
    heatmaps = np.zeros((points_xy.shape[0], out_h, out_w), dtype=np.float32)
    for i, (x, y) in enumerate(points_xy):
        heatmaps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    return np.clip(heatmaps, 0.0, 1.0)

def heatmap_argmax(hm):
    n, h, w = hm.shape
    flat = hm.reshape(n, -1)
    idx = flat.argmax(axis=1)
    ys = (idx // w).astype(np.float32)
    xs = (idx % w).astype(np.float32)
    return np.stack([xs, ys], axis=1)

class SkullHeatmapDataset(Dataset):
    def __init__(self, items, input_size=256, hm_size=64, sigma=1.8, augment=False):
        self.items = items
        self.input_size = int(input_size)
        self.hm_size = int(hm_size)
        self.sigma = float(sigma)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sex, img_path, json_path = self.items[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]

        pts = parse_json_to_slots(json_path)
        if pts is None:
            raise RuntimeError(f"Bad/missing landmarks for slots in: {json_path}")

        if self.augment:
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy()
                pts[:, 0] = (ow - 1) - pts[:, 0]

            if random.random() < 0.35:
                a = random.uniform(-12, 12)
                s = random.uniform(0.95, 1.05)
                cx, cy = ow / 2.0, oh / 2.0
                M = cv2.getRotationMatrix2D((cx, cy), a, s)
                img = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                ones = np.ones((pts.shape[0], 1), dtype=np.float32)
                pts = (np.concatenate([pts, ones], axis=1) @ M.T).astype(np.float32)

        inp = self.input_size
        img_r = cv2.resize(img, (inp, inp), interpolation=cv2.INTER_AREA)
        sx = inp / float(ow)
        sy = inp / float(oh)
        pts_r = pts.copy()
        pts_r[:, 0] *= sx
        pts_r[:, 1] *= sy

        hm = self.hm_size
        pts_hm = pts_r.copy()
        pts_hm[:, 0] *= (hm / float(inp))
        pts_hm[:, 1] *= (hm / float(inp))
        pts_hm[:, 0] = np.clip(pts_hm[:, 0], 0, hm - 1)
        pts_hm[:, 1] = np.clip(pts_hm[:, 1], 0, hm - 1)

        heatmaps = make_heatmaps(pts_hm, hm, hm, self.sigma)

        x = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(heatmaps).float()
        meta = {"sex": sex, "img_path": img_path, "ow": ow, "oh": oh, "inp": inp, "hm": hm}
        return x, y, meta

class ResNetHeatmap(nn.Module):
    def __init__(self, num_points, pretrained=True):
        super().__init__()
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, num_points, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.head(x)
        return x

def overlay_slots(img_bgr, pts_xy, radius=4):
    for i, (x, y) in enumerate(pts_xy):
        lid = int(LANDMARK_SLOTS[i]["id"])
        if lid in SKIP_LANDMARKS:
            continue
        color = LANDMARK_COLORS.get(lid, (255, 255, 255))
        cv2.circle(img_bgr, (int(x), int(y)), radius, color, -1)
    return img_bgr

def run():
    seed = 42
    input_size = 256
    hm_size = 64
    sigma = 1.8
    batch_size = 6
    epochs = 140
    lr = 2e-4
    patience = 20

    set_seed(seed)

    train_items, val_items, _ = split_train_val_test(seed=seed)
    if len(train_items) == 0 or len(val_items) == 0:
        raise RuntimeError("Not enough paired image+json data for train/val. Check filenames and folders.")

    train_ds = SkullHeatmapDataset(train_items, input_size=input_size, hm_size=hm_size, sigma=sigma, augment=True)
    val_ds = SkullHeatmapDataset(val_items, input_size=input_size, hm_size=hm_size, sigma=sigma, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetHeatmap(NUM_LANDMARK_SLOTS, pretrained=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    out_vis = ROOT / "face-reconstruction" / "landmarked_auto_visual"
    out_vis.mkdir(parents=True, exist_ok=True)
    out_model = ROOT / "face-reconstruction" / "landmark_model_heatmap.pth"

    best = float("inf")
    bad = 0

    for epoch in range(epochs):
        model.train()
        tr = 0.0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x)
            if pred.shape[-1] != y.shape[-1]:
                pred = F.interpolate(pred, size=(y.shape[-2], y.shape[-1]), mode="bilinear", align_corners=False)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tr += float(loss.item())

        model.eval()
        va = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                if pred.shape[-1] != y.shape[-1]:
                    pred = F.interpolate(pred, size=(y.shape[-2], y.shape[-1]), mode="bilinear", align_corners=False)
                va += float(loss_fn(pred, y).item())

        tr /= max(1, len(train_loader))
        va /= max(1, len(val_loader))
        print(f"epoch {epoch} train_loss {tr:.6f} val_loss {va:.6f}")

        if va < best - 1e-7:
            best = va
            bad = 0
            torch.save({"state_dict": model.state_dict(), "input_size": input_size, "hm_size": hm_size}, out_model)
        else:
            bad += 1
            if bad >= patience:
                print("early_stop")
                break

        if epoch % 10 == 0:
            x, y, meta = val_ds[random.randrange(len(val_ds))]
            img_path = meta["img_path"]
            sex = meta["sex"]
            img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img0 is not None:
                oh, ow = img0.shape[:2]
                with torch.no_grad():
                    ph = model(x.unsqueeze(0).to(device))[0]
                    ph = torch.sigmoid(ph).detach().cpu().numpy()

                pred_xy_hm = heatmap_argmax(ph)
                pred_xy_inp = pred_xy_hm * (input_size / float(hm_size))
                pred_xy_orig = pred_xy_inp.copy()
                pred_xy_orig[:, 0] *= (ow / float(input_size))
                pred_xy_orig[:, 1] *= (oh / float(input_size))

                gt_xy_hm = heatmap_argmax(y.numpy())
                gt_xy_inp = gt_xy_hm * (input_size / float(hm_size))
                gt_xy_orig = gt_xy_inp.copy()
                gt_xy_orig[:, 0] *= (ow / float(input_size))
                gt_xy_orig[:, 1] *= (oh / float(input_size))

                stem = Path(img_path).stem
                cv2.imwrite(str(out_vis / f"{sex}_{stem}_VAL_pred.jpg"), overlay_slots(img0.copy(), pred_xy_orig, 4))
                cv2.imwrite(str(out_vis / f"{sex}_{stem}_VAL_gt.jpg"), overlay_slots(img0.copy(), gt_xy_orig, 4))

    print("saved_model face-reconstruction/landmark_model_heatmap.pth")
    print("visuals_in face-reconstruction/landmarked_auto_visual/")

if __name__ == "__main__":
    run()
