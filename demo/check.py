import numpy as np
data = np.load("skull_data_split.npz")
print(data.files)
print(data["X_train"].shape, data["y_train"].shape)
print(data["X_test"].shape, data["y_test"].shape)
