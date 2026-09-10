"""Plate Recognition API - Flask + OpenCV + EasyOCR."""

import os
import re

import cv2
import easyocr
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

import detector

load_dotenv()

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "5000"))
FLASK_ENV = os.getenv("FLASK_ENV", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
OCR_GPU = os.getenv("OCR_GPU", "False").strip().lower() in ("1", "true", "yes")

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# Kode wilayah plat Indonesia. Dipakai sebagai filter keras kandidat.
AREA_CODES = {
    "A", "AA", "AB", "AD", "AE", "AG", "B", "BA", "BB", "BD", "BE", "BG", "BH",
    "BK", "BL", "BM", "BN", "BP", "D", "DA", "DB", "DC", "DD", "DE", "DG", "DH",
    "DK", "DL", "DM", "DN", "DP", "DR", "DS", "DT", "DU", "DW", "E", "EA", "EB",
    "ED", "F", "G", "H", "K", "KB", "KH", "KT", "KU", "L", "M", "N", "P", "PA",
    "PB", "R", "S", "T", "W", "Z",
}

# Confusion OCR: dipakai sesuai posisi (huruf vs angka).
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
# Posisi kode wilayah: satu digit bisa jadi beberapa huruf ("0" -> D atau O).
# Kandidat disaring keras oleh AREA_CODES, jadi ambigu di sini aman.
AREA_ALT = {"0": "DO", "1": "IT", "2": "Z", "3": "BE", "4": "A", "5": "S",
            "6": "G", "7": "T", "8": "BR", "9": "P"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else "*")

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
    if w < 640:
        scale = 640 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    yield img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yield cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    yield cv2.threshold(cv2.bilateralFilter(gray, 11, 17, 17), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _coerce(chunk, table):
    """Map tiap char ke kelas target. Return (hasil, jumlah_char_asli) atau None."""
    out, clean = [], 0
    for ch in chunk:
        if (table is TO_DIGIT) == ch.isdigit():
            out.append(ch)
            clean += 1
        elif ch in table:
            out.append(table[ch])
        else:
            return None
    return "".join(out), clean


def _area_candidates(chunk):
    """Semua tafsir kode wilayah yang mungkin dari potongan OCR.

    Digit dipetakan ke beberapa huruf sekaligus (AREA_ALT) karena OCR sering
    membaca "D" sebagai "0". Hasil disaring AREA_CODES oleh pemanggil.
    Return list (teks, jumlah_char_asli).
    """
    outs = [("", 0)]
    for ch in chunk:
        if ch.isalpha():
            opts = [(ch, 1)]
        elif ch in AREA_ALT:
            opts = [(c, 0) for c in AREA_ALT[ch]]
        else:
            return []
        outs = [(p + c, n + k) for p, n in outs for c, k in opts]
    return outs


def _score(area, num, suf, clean, area_clean, num_clean, conf, at_start):
    if area not in AREA_CODES:
        return None
    if num_clean == 0 or num_clean * 2 < len(num):  # angka harus dominan digit asli
        return None
    # Kode wilayah tanpa huruf asli (mis. OCR baca "D" jadi "0") baru diterima
    # kalau ia di awal teks DAN nomornya 4 digit yang semuanya asli. Tanpa itu
    # plat yang kode wilayahnya gagal terbaca ("1002 SJT") akan dicuri digit
    # dan hurufnya jadi "T 0025 JT". Untuk data pajak, not_found lebih aman
    # daripada nomor yang salah.
    if area_clean == 0 and not (at_start and len(num) == 4 and num_clean == 4):
        return None
    return clean + conf * 2 + (2 if suf else 0) + (1 if len(num) == 4 else 0) \
        + (1 if area_clean else 0)


def parse_plate(texts):
    """Plat Indonesia terbaik dari fragmen OCR. Item: str atau (str, confidence)."""
    frags = []
    for t in texts:
        s, c = (t, 1.0) if isinstance(t, str) else (t[0], float(t[1]))
        s = re.sub(r"[^A-Z0-9]", "", s.upper())
        if s:
            frags.append((s, c))
    if not frags:
        return None

    joined = "".join(s for s, _ in frags)
    conf = sum(c for _, c in frags) / len(frags)

    best, best_score = None, 0.0
    n = len(joined)
    for i in range(n):
        for j in range(i + 3, min(i + 10, n + 1)):
            sub = joined[i:j]
            for a in (1, 2):
                for c_len in range(3, -1, -1):
                    b = len(sub) - a - c_len
                    if not 1 <= b <= 4:
                        continue
                    num = _coerce(sub[a:a + b], TO_DIGIT)
                    suf = _coerce(sub[a + b:], TO_LETTER) if c_len else ("", 0)
                    if not (num and suf):
                        continue
                    for area_txt, area_clean in _area_candidates(sub[:a]):
                        clean = area_clean + num[1] + suf[1]
                        sc = _score(area_txt, num[0], suf[0], clean, area_clean,
                                    num[1], conf, i == 0)
                        if sc is not None and sc > best_score:
                            best_score = sc
                            best = " ".join(p for p in (area_txt, num[0], suf[0]) if p)
    return best


def first_line(raw):
    """Ambil fragmen baris nomor plat saja, urut kiri->kanan.

    Plat Indonesia 2 baris: nomor di atas, masa berlaku (BB.YY) di bawah.
    Tanpa pemisahan ini fragmen tanggal ikut tergabung dan parser bisa
    memilih digit tanggal sebagai nomor (mis. "D 5299 UCD" -> "D 12 S").
    Item raw: (bbox, text, conf) dari EasyOCR detail=1.
    """
    if not raw:
        return []
    items = []
    for box, text, conf in raw:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), (max(ys) - min(ys)), min(xs), text, conf))

    tol = max(1.0, sorted(i[1] for i in items)[len(items) // 2] * 0.6)
    top = min(i[0] for i in items)
    line = [i for i in items if i[0] - top < tol]
    line.sort(key=lambda i: i[2])
    return [(i[3], i[4]) for i in line]


def _ocr(img):
    """OCR multi-varian pada satu citra. Berhenti di varian pertama yang valid."""
    reader = get_reader()
    for v in variants(img):
        raw = reader.readtext(v, allowlist=ALLOWLIST, detail=1)
        if not raw:
            continue
        line = first_line(raw)
        plate = parse_plate(line) or parse_plate([(r[1], r[2]) for r in raw])
        if plate:
            return plate, round(sum(c for _, c in line) / len(line), 3)
    return None, 0.0


def read_plate(img):
    """Stage-1 deteksi plat (YOLO) lalu OCR crop-nya saja.

    OCR di crop jauh lebih cepat & akurat daripada di full frame karena
    EasyOCR tidak lagi memindai teks lain (stiker, spanduk, tulisan bak).
    Fallback ke full frame kalau detektor tidak menemukan plat sama sekali.
    """
    try:
        boxes = detector.detect(img)
    except Exception:  # model hilang/korup: jangan matikan endpoint
        boxes = []

    for box in boxes[:3]:  # kandidat teratas saja
        c = detector.crop(img, box)
        if c is None:
            continue
        plate, conf = _ocr(c)
        if plate:
            return plate, conf, box[4]

    plate, conf = _ocr(img)
    return plate, conf, None


@app.get("/health")
def health():
    return jsonify(status="ok", service="plate-recognition-api")


@app.get("/health/ocr")
def health_ocr():
    get_reader()
    return jsonify(status="ready", ocr_engine="easyocr", gpu_enabled=OCR_GPU)


@app.post("/api/detect-plate")
def detect_plate():
    f = request.files.get("file") or request.files.get("image")
    if f is None or not f.filename:
        return jsonify(status="error", message="File gambar wajib diunggah (field: file/image)"), 400
    if f.filename.rsplit(".", 1)[-1].lower() not in ALLOWED_EXT:
        return jsonify(status="error", message=f"Ekstensi tidak didukung. Gunakan: {', '.join(sorted(ALLOWED_EXT))}"), 400

    buf = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(status="error", message="File gambar tidak valid atau rusak"), 400

    plate, conf, det_conf = read_plate(img)
    if plate:
        return jsonify(status="success", plate_number=plate, confidence=conf,
                       detection_confidence=round(det_conf, 3) if det_conf else None)
    return jsonify(status="not_found", message="Plat nomor tidak terdeteksi")


@app.errorhandler(413)
def too_large(_):
    return jsonify(status="error", message="Ukuran file melebihi 10MB"), 413


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=FLASK_ENV == "development")
