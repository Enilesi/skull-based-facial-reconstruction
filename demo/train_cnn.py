import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

# --- SETTINGS ---
base_dir = "."
classes = ["female skulls", "male skulls"]
img_size = (128, 128)

X, y = [], []

# --- LOAD IMAGES ---
for label, folder in enumerate(classes):
    folder_path = os.path.join(base_dir, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size).astype("float32") / 255.0
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)
y_cat = to_categorical(y, num_classes=2)

# --- SPLIT TRAIN/TEST RANDOMLY (balanced) ---
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)
print(f"Train set: {len(X_train)} images, Test set: {len(X_test)} images")

# --- CNN MODEL ---
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- TRAIN ---
history = model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_test, y_test))

# --- EVALUATE ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {acc:.2f}")

# --- SAVE ---
model.save("skull_cnn_model.keras")
print("Model saved as skull_cnn_model.keras")
