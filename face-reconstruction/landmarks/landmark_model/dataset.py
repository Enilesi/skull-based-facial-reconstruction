import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict

import sys
CURRENT_DIR = os.path.dirname(__file__)
LANDMARKS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(LANDMARKS_DIR)

from landmark_slots import LANDMARK_SLOTS, NUM_LANDMARK_SLOTS


class SkullLandmarkDataset(Dataset):
    def __init__(self, landmark_root, split="train"):
        """
        split: "train" or "val"
        Uses:
          - first 3 male + 3 female  -> train
          - next 2 male + 2 female   -> val
        """
        self.samples = []
        samples_by_sex = {"male": [], "female": []}

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
                    continue

                if not os.path.isabs(img_path):
                    img_path = os.path.normpath(
                        os.path.join(os.path.dirname(json_path), img_path)
                    )

                if not os.path.exists(img_path):
                    continue

                samples_by_sex[sex].append((img_path, json_path))

        for sex in ["male", "female"]:
            if split == "train":
                self.samples += samples_by_sex[sex][:3]
            else:  # val
                self.samples += samples_by_sex[sex][3:5]

        print(f"[INFO] Dataset ({split}) loaded with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lmk_path = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot load image {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        with open(lmk_path) as f:
            data = json.load(f)

        by_id = defaultdict(list)
        for p in data["landmarks"]:
            by_id[p["id"]].append(p)

        coords = []

        for slot in LANDMARK_SLOTS:
            lid = slot["id"]
            side = slot["side"]

            pts = by_id.get(lid)
            if pts is None:
                raise ValueError(f"Missing landmark id {lid} in {lmk_path}")

            if side == "mid":
                p = pts[0]
            else:
                pts_sorted = sorted(pts, key=lambda p: p["x"])
                p = pts_sorted[0] if side == "left" else pts_sorted[1]

            coords.append(p["x"] / w)
            coords.append(p["y"] / h)

        coords = torch.tensor(coords, dtype=torch.float32)

        if coords.numel() != NUM_LANDMARK_SLOTS * 2:
            raise RuntimeError(
                f"Expected {NUM_LANDMARK_SLOTS*2} values, got {coords.numel()}"
            )

        return img, coords
