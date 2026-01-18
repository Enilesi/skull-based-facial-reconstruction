import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from landmarks.landmark_slots import LANDMARK_SLOTS, NUM_LANDMARK_SLOTS
from landmarks.landmarks_schema import SKIP_LANDMARKS

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

class ResNetHeatmap(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True))
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

def heatmap_argmax(hm):
    n, h, w = hm.shape
    flat = hm.reshape(n, -1)
    idx = flat.argmax(axis=1)
    ys = (idx // w).astype(np.float32)
    xs = (idx % w).astype(np.float32)
    return np.stack([xs, ys], axis=1)

def overlay_slots(img_bgr, pts_xy, radius=4):
    for i, (x, y) in enumerate(pts_xy):
        lid = int(LANDMARK_SLOTS[i]["id"])
        if lid in SKIP_LANDMARKS:
            continue
        color = LANDMARK_COLORS.get(lid, (255, 255, 255))
        cv2.circle(img_bgr, (int(x), int(y)), radius, color, -1)
    return img_bgr

def load_json_presence():
    root = ROOT / "data" / "landmarked"
    present = set()
    for sex, folder in SEX_FOLDERS:
        p = root / folder
        if not p.exists():
            continue
        for j in p.glob("*_landmarks.json"):
            img_id = j.stem.replace("_landmarks", "")
            present.add((sex, img_id))
    return present

def collect_test_images():
    root = ROOT / "data" / "images"
    present = load_json_presence()
    out = []
    for sex, _ in SEX_FOLDERS:
        p = root / sex
        if not p.exists():
            continue
        for fp in p.rglob("*"):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in IMAGE_EXTS:
                continue
            img_id = fp.stem
            if (sex, img_id) not in present:
                out.append((sex, str(fp)))
    return out

def main():
    ckpt_path = ROOT / "face-reconstruction" / "landmark_model_heatmap.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_size = int(ckpt["input_size"])
    hm_size = int(ckpt["hm_size"])

    model = ResNetHeatmap(NUM_LANDMARK_SLOTS)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir = ROOT / "face-reconstruction" / "landmarked_auto_visual"
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = collect_test_images()
    print(f"Found {len(tests)} test images.")

    for sex, img_path in tests:
        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img0 is None:
            continue
        oh, ow = img0.shape[:2]

        rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb_r).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            ph = model(x)[0]
            ph = torch.sigmoid(ph).cpu().numpy()

        pred_xy_hm = heatmap_argmax(ph)
        pred_xy_inp = pred_xy_hm * (input_size / float(hm_size))
        pred_xy_orig = pred_xy_inp.copy()
        pred_xy_orig[:, 0] *= (ow / float(input_size))
        pred_xy_orig[:, 1] *= (oh / float(input_size))

        out_img = overlay_slots(img0.copy(), pred_xy_orig, 4)
        stem = Path(img_path).stem
        cv2.imwrite(str(out_dir / f"{sex}_{stem}_TEST_pred.jpg"), out_img)

    print("visuals_in face-reconstruction/landmarked_auto_visual/")

if __name__ == "__main__":
    main()
