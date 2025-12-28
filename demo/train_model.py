import os
import cv2
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- SETTINGS ---
base_dir = "."
classes = ["female skulls", "male skulls"]
img_size = (128, 128)

X_train, y_train, X_test, y_test = [], [], [], []

# --- LOAD AND SPLIT IMAGES ---
for label, folder in enumerate(classes):
    folder_path = os.path.join(base_dir, folder)
    images = sorted(os.listdir(folder_path))
    train_imgs = images[:-5]
    test_imgs = images[-5:]

    for img_name in train_imgs:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size).astype("float32") / 255.0
        X_train.append(img.flatten())
        y_train.append(label)

    for img_name in test_imgs:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size).astype("float32") / 255.0
        X_test.append(img.flatten())
        y_test.append(label)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"Train set: {len(X_train)} images, Test set: {len(X_test)} images")
print("Train labels count:", np.bincount(y_train))
print("Test labels count:", np.bincount(y_test))

# --- TRAIN MODEL ---
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("\nModel trained successfully.")
print(f"Train accuracy: {train_acc:.2f}")
print(f"Test accuracy: {test_acc:.2f}")
print("\nClassification report:\n", classification_report(y_test, model.predict(X_test), target_names=classes))

# --- SAVE MODEL ---
joblib.dump(model, "skull_softmax_model.pkl")
np.savez("skull_data_split_normalized.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("\nModel and data saved.")
