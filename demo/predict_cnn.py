import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

if len(sys.argv) < 2:
    print("Usage: python predict_cnn.py <image_path>")
    exit(1)

image_path = sys.argv[1]
model = load_model("skull_cnn_model.keras")

img = cv2.imread(image_path)
if img is None:
    print(f"Image '{image_path}' not found.")
    exit(1)

img = cv2.resize(img, (128, 128)).astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0]
classes = ["Female skull", "Male skull"]
label = classes[np.argmax(pred)]

print(f"Prediction for '{image_path}': {label}")
