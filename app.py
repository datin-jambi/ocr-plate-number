"""Plate Recognition API - Flask. Lapisan HTTP saja; logika ada di pipeline.py."""

import os

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from app.pipeline import OCR_GPU, get_reader, read_plate

load_dotenv()

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "5000"))
FLASK_ENV = os.getenv("FLASK_ENV", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else "*")


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
