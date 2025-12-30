import cv2
import numpy as np
import requests

IMAGE_PATH = "skull-male.png"
OUT_IMAGE_PATH = "skull_with_landmarks.png"
API_URL = "http://127.0.0.1:8000/predict"

REAL_XCB_MM = 140.0  # calibration reference

labels = [
    "EuL", "EuR",
    "ZyL", "ZyR",
    "FmtL", "FmtR",
    "MfL", "MfR",
    "AlL", "AlR",
]

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
        points.append((x, y))
        print(f"{labels[len(points)-1]}: {x}, {y}")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("Could not load image")

img_display = img.copy()

cv2.imshow("Click landmarks in order", img_display)
cv2.setMouseCallback("Click landmarks in order", click_event)

while True:
    cv2.imshow("Click landmarks in order", img_display)
    if (cv2.waitKey(1) & 0xFF) == 27 or len(points) == len(labels):
        break

cv2.destroyAllWindows()

debug_img = img.copy()
for (x, y), name in zip(points, labels):
    cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(debug_img, name, (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

cv2.imwrite(OUT_IMAGE_PATH, debug_img)

(EuL, EuR,
 ZyL, ZyR,
 FmtL, FmtR,
 MfL, MfR,
 AlL, AlR) = points

def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

XCB_px = dist(EuL, EuR)
mm_per_px = REAL_XCB_MM / XCB_px

measurements = {
    "XCB": XCB_px * mm_per_px,
    "ZYB": dist(ZyL, ZyR) * mm_per_px,
    "WFB": dist(FmtL, FmtR) * mm_per_px,
    "OBBL": dist(MfL, MfR) * mm_per_px,
    "NLB": dist(AlL, AlR) * mm_per_px,

    # Not recoverable from single 2D image
    "BBH": None,
    "BNL": None,
    "BPL": None,
    "FRC": None,
    "GOL": None,
    "MAB": None,
    "NLHL": None,
    "OBHL": None,
    "OCC": None,
    "PAC": None,
    "UFHT": None,
    "X": None,
    "Y": None,
    "Z": None,
    "W": None
}

print("\nMeasurements (mm, approximated):")
for k, v in measurements.items():
    print(f"{k}: {v if v is not None else 'N/A'}")

payload = {
    "method": "tabular",
    "source": "2D_image_approximation",
    "calibration": "XCB",
    "measurements": measurements
}

r = requests.post(API_URL, json=payload)
print("\nPrediction:")
print(r.json())
