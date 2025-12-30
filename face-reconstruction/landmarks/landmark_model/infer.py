import torch
import cv2
import json
import os
from model import LandmarkCNN
import numpy as np

IMAGE_ROOT = "../../data/images"
OUT_DIR = "../landmarked_auto"
MODEL_PATH = "landmark_model.pth"

os.makedirs(OUT_DIR, exist_ok=True)

# Load model
dummy_landmarks = 42  # update if needed
model = LandmarkCNN(dummy_landmarks)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def process(sex):
    img_dir = os.path.join(IMAGE_ROOT, sex)
    out_dir = os.path.join(OUT_DIR, sex)
    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".png",".jpg",".jpeg",".webp")):
            continue

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        img_r = cv2.resize(img, (224,224))
        img_r = img_r.astype(np.float32)/255.0
        img_t = torch.from_numpy(img_r).permute(2,0,1).unsqueeze(0)

        with torch.no_grad():
            pred = model(img_t).squeeze().numpy()

        landmarks = []
        for i in range(0, len(pred), 2):
            x = int(pred[i] * w)
            y = int(pred[i+1] * h)
            landmarks.append({"id": i//2 + 1, "x": x, "y": y})

        out_json = os.path.join(out_dir, img_name.rsplit(".",1)[0] + "_auto_landmarks.json")
        with open(out_json, "w") as f:
            json.dump({"image": img_path, "sex": sex, "landmarks": landmarks}, f, indent=2)

        print(f"[AUTO] {img_path}")

process("male")
process("female")
