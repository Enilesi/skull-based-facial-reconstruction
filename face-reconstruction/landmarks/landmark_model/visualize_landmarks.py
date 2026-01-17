import os
import sys
import json
import cv2

CURRENT_DIR = os.path.dirname(__file__)
LANDMARKS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(LANDMARKS_DIR)

from landmark_colors import LANDMARK_COLORS

LANDMARKED_AUTO = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../landmarked_auto")
)

OUTPUT_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../landmarked_auto_visual")
)

WINDOW_NAME = "Landmark Visualization"
SAVE_IMAGES = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_landmarks(img, landmarks):
    for p in landmarks:
        lid = p["id"]
        x, y = int(p["x"]), int(p["y"])

        if lid not in LANDMARK_COLORS:
            continue  

        color_name, color = LANDMARK_COLORS[lid]

        cv2.circle(img, (x, y), 5, color, -1)
        cv2.putText(
            img,
            str(lid),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )
    return img


def visualize_folder(sex):
    folder = os.path.join(LANDMARKED_AUTO, sex)
    out_dir = os.path.join(OUTPUT_DIR, sex)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}")
        return

    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith("_auto_landmarks.json")
    )

    print(f"[INFO] Visualizing {len(files)} {sex} skulls")

    for file in files:
        json_path = os.path.join(folder, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        img_path = data["image"]
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Cannot load image {img_path}")
            continue

        img_vis = img.copy()
        img_vis = draw_landmarks(img_vis, data["landmarks"])

        cv2.imshow(WINDOW_NAME, img_vis)

        if SAVE_IMAGES:
            out_path = os.path.join(
                out_dir,
                os.path.basename(img_path).rsplit(".", 1)[0] + "_vis.png"
            )
            cv2.imwrite(out_path, img_vis)

        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_folder("female")
    visualize_folder("male")
