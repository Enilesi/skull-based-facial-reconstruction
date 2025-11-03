import sys, os, cv2, numpy as np, torch
from segment_anything import sam_model_registry, SamPredictor

if len(sys.argv) < 2:
    print("usage: python detect_orbits_gender.py path/to/skull.jpg")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print("❌ File not found:", img_path)
    sys.exit(1)

image = cv2.imread(img_path)
if image is None:
    print("❌ Could not load image:", img_path)
    sys.exit(1)

h, w = image.shape[:2]
checkpoint = "sam_vit_h_4b8939.pth"
if not os.path.exists(checkpoint):
    print("❌ Missing model checkpoint.")
    print("Download it via:")
    print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=checkpoint).to(device)
predictor = SamPredictor(sam)
predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

regions = {
    "left":  [int(w*0.24), int(h*0.30), int(w*0.46), int(h*0.56)],
    "right": [int(w*0.54), int(h*0.30), int(w*0.76), int(h*0.56)],
}

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
output_masks = []
combined_mask = np.zeros((h, w), np.uint8)

for x1, y1, x2, y2 in regions.values():
    box = np.array([x1, y1, x2, y2])
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)].astype(np.uint8) * 255
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 41, 5)
    refined = cv2.bitwise_and(best_mask, dark)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=3)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.dilate(refined, kernel, iterations=2)
    output_masks.append(refined)
    combined_mask = cv2.bitwise_or(combined_mask, refined)

combined_mask = cv2.GaussianBlur(combined_mask, (5,5), 0)
combined_mask = cv2.threshold(combined_mask, 128, 255, cv2.THRESH_BINARY)[1]
overlay = image.copy()
overlay[combined_mask > 0] = [0, 255, 0]

results = {}
for side, mask in zip(regions.keys(), output_masks):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        results[side] = "unknown"
        continue
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w_box, h_box = cv2.boundingRect(c)
    aspect_ratio = w_box / float(h_box)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    extent = area / (w_box * h_box + 1e-6)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    rect = cv2.minAreaRect(c)
    rect_w, rect_h = rect[1]
    rectangularity = area / (rect_w * rect_h + 1e-6)
    aspect_ratio = w_box / float(h_box)

    female_score = 0
    male_score = 0

    if circularity >= rectangularity: female_score += 1
    else: male_score += 1

    if aspect_ratio <= 1.15: female_score += 1
    else: male_score += 1

    if rectangularity <= 0.55: female_score += 1
    else: male_score += 1

    if female_score > male_score:
        gender = "female"
    elif male_score > female_score:
        gender = "male"
    else:
        gender = "uncertain"

    print(f"[{side}] aspect={aspect_ratio:.2f}, circ={circularity:.2f}, rect={rectangularity:.2f}, extent={extent:.2f} → {gender}")

    results[side] = gender
    cv2.putText(overlay, f"{side}: {gender}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

base = os.path.splitext(os.path.basename(img_path))[0]
cv2.imwrite(f"{base}_orbit_mask_final.png", combined_mask)
cv2.imwrite(f"{base}_orbit_overlay_final.png", overlay)
print("✅ Saved final refined masks:")
print(f" - {base}_orbit_mask_final.png")
print(f" - {base}_orbit_overlay_final.png")
print("\n--- Eye socket classification ---")
for side, gender in results.items():
    print(f"{side.capitalize()} orbit → {gender}")
