import cv2
import json
import os

from landmarks_schema import LANDMARKS, SKIP_LANDMARKS
from landmark_colors import LANDMARK_COLORS

BASE_IMAGE_DIR = "../../data/images"
OUTPUT_DIR = "../landmarked"

os.makedirs(OUTPUT_DIR, exist_ok=True)


PLACEMENT_SEQUENCE = []

for lid in sorted(LANDMARKS.keys()):
    if lid in SKIP_LANDMARKS:
        continue
    _, _, count = LANDMARKS[lid]
    for i in range(count):
        PLACEMENT_SEQUENCE.append(lid)


LANDMARK_IDS = PLACEMENT_SEQUENCE

print("\nLANDMARK COLOR MAP:")
for lid in LANDMARK_IDS:
    name, color = LANDMARK_COLORS[lid]
    print(f"Landmark {lid:2d} -> {name:12s} -> BGR {color}")
print("\nClick landmarks in THIS ORDER:")
for lid in LANDMARK_IDS:
    print(f"{lid}: {LANDMARKS[lid][0]}")

current_points = []
current_index = 0
img = None
img_display = None
img_original = None 

def redraw_image():
    global img_display
    img_display = img_original.copy()
    for p in current_points:
        lid = p["id"]
        color_name, color = LANDMARK_COLORS[lid]
        x, y = p["x"], p["y"]

        cv2.circle(img_display, (x, y), 5, color, -1)
        cv2.putText(
            img_display,
            str(lid),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

def click_event(event, x, y, flags, param):
    global current_index, current_points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if current_index >= len(LANDMARK_IDS):
        return

    lid = LANDMARK_IDS[current_index]
    lname = LANDMARKS[lid][0]
    color_name, color = LANDMARK_COLORS[lid]

    current_points.append({
        "id": lid,
        "name": lname,
        "x": int(x),
        "y": int(y)
    })

    current_index += 1
    redraw_image()

    print(f"Placed landmark {lid} ({lname}) at ({x},{y}) using color {color_name}")

def annotate_image(image_path, sex):
    global img, img_display, img_original, current_points, current_index

    current_points = []
    current_index = 0

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot load {image_path}")
        return

    img_original = img.copy()   # ← NEW
    img_display = img.copy()

    cv2.namedWindow("Annotate Skull", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate Skull", click_event)

    while True:
        cv2.imshow("Annotate Skull", img_display)
        key = cv2.waitKey(20) & 0xFF

        # UNDO last point (Z)
        if key == ord('z') or key == ord('Z'):
            if current_points:
                removed = current_points.pop()
                current_index -= 1
                redraw_image()
                print(f"Undo landmark {removed['id']} ({removed['name']})")

        # FINISH after all landmarks placed
        if key != 255 and current_index >= len(LANDMARK_IDS):
            break

    cv2.destroyAllWindows()

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_img = os.path.join(OUTPUT_DIR, f"{base}_landmarked.png")
    out_json = os.path.join(OUTPUT_DIR, f"{base}_landmarks.json")

    cv2.imwrite(out_img, img_display)

    with open(out_json, "w") as f:
        json.dump(
            {
                "image": image_path,
                "sex": sex,
                "landmarks": current_points
            },
            f,
            indent=2
        )

    print(f"[SAVED] {out_img}")
    print(f"[SAVED] {out_json}\n")

def run():
    for sex in ["female", "male"]:
        folder = os.path.join(BASE_IMAGE_DIR, sex)
        if not os.path.isdir(folder):
            continue

        for file in sorted(os.listdir(folder)):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder, file)
                print(f"\nAnnotating {image_path}")
                annotate_image(image_path, sex)

if __name__ == "__main__":
    run()
