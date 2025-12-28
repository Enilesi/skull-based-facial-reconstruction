import numpy as np
import cv2
import joblib
from sklearn.linear_model import LogisticRegression

# Load prepared data
data = np.load("skull_data_split.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Train a Softmax Regression model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully.")
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# Save the model
joblib.dump(model, "skull_softmax_model.pkl")
print("Model saved as skull_softmax_model.pkl")

# --- Prediction for a new image ---
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return
    img = cv2.resize(img, (128, 128))
    img_flat = img.flatten().reshape(1, -1)
    pred = model.predict(img_flat)[0]
    label = "Female skull" if pred == 0 else "Male skull"
    print(f"Prediction for '{image_path}': {label}")

# Example usage
predict_image("test2.jpg")  # uncomment and change this path to test a new image
