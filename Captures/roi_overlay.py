import cv2
# --- edit these ---
IMAGE_PATH = "pov_test.png"
ROI = (210, 260, 160, 45)  # x, y, w, h
OUT_PATH = "ROI_test_image.png"
# -------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH}")
x, y, w, h = ROI
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
cv2.imwrite(OUT_PATH, img)
print(f"Saved: {OUT_PATH}")