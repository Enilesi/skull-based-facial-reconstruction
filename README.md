https://data.mendeley.com/datasets/byr94xy7mv/1

Milestone 1 — Sex from measurements (you already have)

✅ Trained model on ILDs, working /predict endpoint.

Milestone 2 — Frontal image → landmarks (you will build)

You will create a small landmark labeling set on your skull images.

Milestone 3 — Landmarks → ILDs (script)

Convert landmark coordinates into the same kinds of distances the model expects.

Milestone 4 — Image → sex (end-to-end)

Frontal skull image → predicted sex.

Milestone 5 — Face approximation (basic reconstruction)

Use sex + skull geometry cues to generate a probabilistic facial approximation (not identity).







1️⃣ Zygion Left (ZyL)

The widest point of the left cheekbone

Look at where the face is widest at cheek level

Click the outermost point, not the eye socket

2️⃣ Zygion Right (ZyR)

Same as ZyL, but on the right side

➡️ Distance ZyL–ZyR = ZYB

3️⃣ Euryon Left (EuL)

The widest point of the skull on the left

Higher than the cheekbones

Usually near the parietal bone bulge

4️⃣ Euryon Right (EuR)

Same as EuL, but on the right

➡️ Distance EuL–EuR = XCB

Using additional landmarks:
5. G – glabella (brow ridge midline)
6. N – nasion (nose root)
7. OrL – left orbitale (lowest orbital point)
8. OrR – right orbitale
9. AlL – left alare (nose widest point)
10. AlR – right alare

