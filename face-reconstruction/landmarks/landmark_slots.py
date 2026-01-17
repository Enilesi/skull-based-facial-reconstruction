from landmarks_schema import LANDMARKS, SKIP_LANDMARKS

LANDMARK_SLOTS = []

for lid, (name, _, count) in LANDMARKS.items():
    if lid in SKIP_LANDMARKS:
        continue

    if count == 1:
        LANDMARK_SLOTS.append({
            "id": lid,
            "name": name,
            "side": "mid"
        })
    else:
        LANDMARK_SLOTS.append({
            "id": lid,
            "name": name,
            "side": "left"
        })
        LANDMARK_SLOTS.append({
            "id": lid,
            "name": name,
            "side": "right"
        })

NUM_LANDMARK_SLOTS = len(LANDMARK_SLOTS)  # MUST be 30
