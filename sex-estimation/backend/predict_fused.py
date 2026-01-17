import subprocess
import json

LANDMARK_PREDICT_SCRIPT = "click_landmarks.py"
IMAGE_PREDICT_SCRIPT = "predict_image.py"

LANDMARK_WEIGHT = 0.3
IMAGE_WEIGHT = 0.7

print("Running landmark-based prediction...")
lm_proc = subprocess.run(
    ["python", LANDMARK_PREDICT_SCRIPT],
    capture_output=True,
    text=True
)

lm_out = lm_proc.stdout
print(lm_out)

def extract_prob(text):
    for line in text.splitlines():
        if "male_probability" in line:
            return float(line.split(":")[-1].strip().replace("}", ""))
    return None

p_landmark = extract_prob(lm_out)
if p_landmark is None:
    raise RuntimeError("Could not extract landmark probability")

print("\nRunning image-based prediction...")
img_proc = subprocess.run(
    ["python", IMAGE_PREDICT_SCRIPT],
    capture_output=True,
    text=True
)

img_out = img_proc.stdout
print(img_out)

for line in img_out.splitlines():
    if "Male probability" in line:
        p_image = float(line.split(":")[-1].strip())

p_final = LANDMARK_WEIGHT * p_landmark + IMAGE_WEIGHT * p_image

if p_final >= 0.55:
    sex = "M"
elif p_final <= 0.5:
    sex = "F"
else:
    sex = "Ambiguous"


print("\n==============================")
print("FINAL FUSED RESULT")
print("==============================")
print(f"Landmark male prob : {p_landmark:.3f}")
print(f"Image male prob    : {p_image:.3f}")
print(f"Final male prob    : {p_final:.3f}")
print(f"Decision           : {sex}")
