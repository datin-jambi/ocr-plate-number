"""Pipeline pengenalan plat: deteksi (YOLO) -> crop -> OCR -> parse.

Menyatukan detector (stage-1) dan EasyOCR (stage-2). Tidak tahu apa-apa
soal HTTP; app.py yang membungkusnya jadi endpoint.
"""

import os

import cv2
import easyocr

from app import detector
from app.plates import first_line, parse_plate

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_GPU = os.getenv("OCR_GPU", "False").strip().lower() in ("1", "true", "yes")
MIN_OCR_WIDTH = 640  # crop plat biasanya kecil; perbesar dulu agar teks terbaca
MAX_CANDIDATES = 3   # kotak plat teratas yang dicoba sebelum menyerah

_reader = None


def get_reader():
    """Lazy singleton so the model loads once per worker process."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=OCR_GPU)
    return _reader


def variants(img):
    """Beberapa versi gambar; EasyOCR paling akurat di citra natural, bukan biner."""
    h, w = img.shape[:2]
    if w < MIN_OCR_WIDTH:
        scale = MIN_OCR_WIDTH / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    yield img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yield cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    yield cv2.threshold(cv2.bilateralFilter(gray, 11, 17, 17), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _ocr(img):
    """OCR multi-varian pada satu citra. Berhenti di varian pertama yang valid."""
    reader = get_reader()
    for v in variants(img):
        raw = reader.readtext(v, allowlist=ALLOWLIST, detail=1)
        if not raw:
            continue
        line = first_line(raw)
        # fallback ke semua fragmen kalau pemisahan baris terlalu agresif
        plate = parse_plate(line) or parse_plate([(r[1], r[2]) for r in raw])
        if plate:
            return plate, round(sum(c for _, c in line) / len(line), 3)
    return None, 0.0


def read_plate(img):
    """Stage-1 deteksi plat (YOLO) lalu OCR crop-nya saja.

    OCR di crop jauh lebih cepat & akurat daripada di full frame karena
    EasyOCR tidak lagi memindai teks lain (stiker, spanduk, tulisan bak).
    Fallback ke full frame kalau detektor tidak menemukan plat sama sekali.
    Return (plat, confidence_ocr, confidence_deteksi | None).
    """
    try:
        boxes = detector.detect(img)
    except Exception:  # model hilang/korup: jangan matikan endpoint
        boxes = []

    for box in boxes[:MAX_CANDIDATES]:
        crop = detector.crop(img, box)
        if crop is None:
            continue
        plate, conf = _ocr(crop)
        if plate:
            return plate, conf, box[4]

    plate, conf = _ocr(img)
    return plate, conf, None
