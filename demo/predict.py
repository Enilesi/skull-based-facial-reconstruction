import sys
import cv2
import numpy as np
import joblib

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    exit(1)

image_path = sys.argv[1]

# --- LOAD MODEL ---
model = joblib.load("skull_softmax_model.pkl")

# --- PREPROCESS IMAGE ---
img = cv2.imread(image_path)
if img is None:
    print(f"Image '{image_path}' not found.")
    exit(1)

img = cv2.resize(img, (128, 128)).astype("float32") / 255.0
img_flat = img.flatten().reshape(1, -1)

# --- PREDICT ---
pred = model.predict(img_flat)[0]
label = "Female skull" if pred == 0 else "Male skull"

print(f"\nPrediction for '{image_path}': {label}")
