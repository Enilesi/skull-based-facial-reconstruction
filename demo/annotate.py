import cv2, csv, os

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("image", img)

folder = "male skulls"
output = "skull_points.csv"

with open(output, "a", newline="") as f:
    writer = csv.writer(f)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        writer.writerow([filename] + [coord for p in points for coord in p])
        points.clear()
print("Annotation completed and saved to", output)
