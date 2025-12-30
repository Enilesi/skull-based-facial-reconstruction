import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class SkullLandmarkDataset(Dataset):
    def __init__(self, landmark_root):
        self.samples = []

        for sex in ["male", "female"]:
            lmk_dir = os.path.join(landmark_root, f"landmarked_{sex}")
            if not os.path.isdir(lmk_dir):
                continue

            for file in sorted(os.listdir(lmk_dir)):
                if not file.endswith("_landmarks.json"):
                    continue

                json_path = os.path.join(lmk_dir, file)

                with open(json_path) as f:
                    data = json.load(f)

                img_path = data.get("image")
                if img_path is None:
                    print(f"[WARN] No image field in {json_path}")
                    continue

                # 🔑 CRITICAL FIX: resolve relative paths correctly
                if not os.path.isabs(img_path):
                    img_path = os.path.normpath(
                        os.path.join(os.path.dirname(json_path), img_path)
                    )

                if not os.path.exists(img_path):
                    print(f"[WARN] Image not found: {img_path}")
                    continue

                self.samples.append((img_path, json_path))

        print(f"[INFO] Dataset loaded with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lmk_path = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        with open(lmk_path) as f:
            data = json.load(f)

        coords = []
        for p in data["landmarks"]:
            coords.append(p["x"] / w)
            coords.append(p["y"] / h)

        coords = torch.tensor(coords, dtype=torch.float32)

        return img, coords
