"""Buat gambar plat sintetis untuk smoke test endpoint."""
import cv2
import numpy as np

img = np.full((160, 520, 3), 20, np.uint8)
cv2.rectangle(img, (10, 10), (510, 150), (255, 255, 255), -1)
cv2.putText(img, "B 1234 ABC", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (10, 10, 10), 6)
cv2.imwrite("sample_plate.jpg", img)
print("wrote sample_plate.jpg")
