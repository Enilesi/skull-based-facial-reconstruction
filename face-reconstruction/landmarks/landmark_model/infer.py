import os
import json
import cv2
import torch
import numpy as np

from model import LandmarkCNN

import sys
CURRENT_DIR = os.path.dirname(__file__)
LANDMARKS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(LANDMARKS_DIR)

from landmark_slots import LANDMARK_SLOTS, NUM_LANDMARK_SLOTS

IMAGE_ROOT = "../../../data/images"
OUTPUT_ROOT = "../../landmarked_auto"
MODEL_PATH = "landmark_model.pth"
SKIP_FIRST_N = 5

os.makedirs(OUTPUT_ROOT, exist_ok=True)
model = LandmarkCNN(NUM_LANDMARK_SLOTS)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def preprocess(img):
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img, h, w

def run(sex):
    img_dir = os.path.join(IMAGE_ROOT, sex)
    out_dir = os.path.join(OUTPUT_ROOT, sex)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )

    for i, fname in enumerate(files):
        if i < SKIP_FIRST_N:
            continue

        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_t, h, w = preprocess(img)

        with torch.no_grad():
            pred = model(img_t).squeeze().numpy()

        landmarks = []
        idx = 0
        for slot in LANDMARK_SLOTS:
            x = int(pred[idx] * w)
            y = int(pred[idx + 1] * h)
            idx += 2

            landmarks.append({
                "id": slot["id"],
                "x": x,
                "y": y
            })

        out_json = os.path.join(
            out_dir,
            fname.rsplit(".", 1)[0] + "_auto_landmarks.json"
        )

        with open(out_json, "w") as f:
            json.dump(
                {
                    "image": img_path,
                    "sex": sex,
                    "landmarks": landmarks,
                    "source": "cnn_auto"
                },
                f,
                indent=2
            )

        print(f"[AUTO] {sex}: {fname}")

run("male")
run("female")
