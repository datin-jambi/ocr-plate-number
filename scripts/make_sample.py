"""Buat gambar plat sintetis untuk smoke test endpoint.

Jalankan: python scripts/make_sample.py
Output: sample_plate.jpg di direktori project root.
"""
import os
import sys

import cv2
import numpy as np

# Output ke root project, bukan ke scripts/
output_path = os.path.join(os.path.dirname(__file__), "..", "sample_plate.jpg")

img = np.full((160, 520, 3), 20, np.uint8)
cv2.rectangle(img, (10, 10), (510, 150), (255, 255, 255), -1)
cv2.putText(img, "B 1234 ABC", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (10, 10, 10), 6)
cv2.imwrite(output_path, img)
print(f"wrote {os.path.normpath(output_path)}")
