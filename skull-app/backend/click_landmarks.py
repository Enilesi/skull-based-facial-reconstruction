import cv2
import numpy as np
import requests

IMAGE_PATH = "img.jpg"
OUT_IMAGE_PATH = "skull_with_landmarks.png"
API_URL = "http://127.0.0.1:8000/predict"

labels = [
    "EuL", "EuR",
    "ZyL", "ZyR",
    "FmtL", "FmtR",
    "OrL", "OrR",
    "AlL", "AlR"
]

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"{labels[len(points)-1]}: {x}, {y}")

img = cv2.imread(IMAGE_PATH)
img_display = img.copy()

cv2.imshow("Click landmarks in order", img_display)
cv2.setMouseCallback("Click landmarks in order", click_event)

while True:
    cv2.imshow("Click landmarks in order", img_display)
    if cv2.waitKey(1) & 0xFF == 27 or len(points) == len(labels):
        break

cv2.destroyAllWindows()

debug_img = img.copy()
for (x, y), name in zip(points, labels):
    cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(debug_img, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imwrite(OUT_IMAGE_PATH, debug_img)
print(f"\nSaved debug image: {OUT_IMAGE_PATH}")

(EuL, EuR, ZyL, ZyR, FmtL, FmtR, OrL, OrR, AlL, AlR) = points

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

XCB_px = dist(EuL, EuR)
ZYB_px = dist(ZyL, ZyR)
WFB_px = dist(FmtL, FmtR)
OBBL_px = dist(OrL, OrR)
NLB_px = dist(AlL, AlR)

measurements = {
    "ZYB_XCB": ZYB_px / XCB_px,
    "WFB_XCB": WFB_px / XCB_px,
    "OBBL_XCB": OBBL_px / XCB_px,
    "NLB_XCB": NLB_px / XCB_px,
    "NLB_ZYB": NLB_px / ZYB_px
}

print("\nRatios:")
for k, v in measurements.items():
    print(f"{k}: {v:.4f}")

r = requests.post(API_URL, json={"measurements": measurements})
print("\nPrediction:")
print(r.json())
