import os
import cv2
import numpy as np

# Paths
base_dir = "."
classes = ["female skulls", "male skulls"]

# Parameters
img_size = (128, 128)  # resize for consistency

X_train, y_train, X_test, y_test = [], [], [], []

for label, folder in enumerate(classes):
    folder_path = os.path.join(base_dir, folder)
    images = sorted(os.listdir(folder_path))
    
    # Split: all except last 5 -> train, last 5 -> test
    train_imgs = images[:-5]
    test_imgs = images[-5:]
    
    for img_name in train_imgs:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        X_train.append(img.flatten())
        y_train.append(label)
    
    for img_name in test_imgs:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        X_test.append(img.flatten())
        y_test.append(label)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Train set: {len(X_train)} images")
print(f"Test set: {len(X_test)} images")

# Optional: save arrays for later use
np.savez("skull_data_split.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("Data saved to skull_data_split.npz")
