import os
import json

# We are in:
# skull-based-facial-reconstruction/face-reconstruction/landmarks/
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_IMAGES = os.path.join(PROJECT_ROOT, "data", "images")
LANDMARKED_ROOT = os.path.join(PROJECT_ROOT, "face-reconstruction", "landmarked")

def fix_folder(sex):
    folder = os.path.join(LANDMARKED_ROOT, f"landmarked_{sex}")

    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return

    for file in os.listdir(folder):
        if not file.endswith("_landmarks.json"):
            continue

        json_path = os.path.join(folder, file)

        with open(json_path) as f:
            data = json.load(f)

        old_path = data.get("image")
        if old_path is None:
            print(f"[WARN] No image field in {json_path}")
            continue

        image_name = os.path.basename(old_path)
        new_path = os.path.join(DATA_IMAGES, sex, image_name)

        if not os.path.exists(new_path):
            print(f"[WARN] Image not found: {new_path}")
            continue

        data["image"] = new_path

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[FIXED] {json_path}")

fix_folder("male")
fix_folder("female")
