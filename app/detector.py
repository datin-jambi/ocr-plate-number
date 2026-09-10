"""Stage-1: deteksi kotak plat pakai YOLO11n ONNX (onnxruntime, CPU).

Output model: [1, 5, N] = (cx, cy, w, h, conf) dalam skala 640x640 letterbox.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.getenv("PLATE_MODEL", "models/license-plate.onnx")
IMG_SIZE = 640
CONF_THRES = float(os.getenv("PLATE_CONF", "0.35"))
IOU_THRES = 0.45
PAD_RATIO = 0.08  # crop dilebarkan sedikit agar karakter tepi tidak terpotong

_session = None


def get_session():
    """Lazy singleton; model dimuat sekali per proses worker."""
    global _session
    if _session is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.getenv("PLATE_THREADS", "2"))
        _session = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    return _session


def _letterbox(img):
    """Resize jaga aspek ratio + pad abu-abu. Return (blob, ratio, pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, np.uint8)
    dx, dy = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), r, dx, dy


def detect(img):
    """Return list kotak plat [(x1, y1, x2, y2, conf)], urut confidence menurun."""
    blob, r, dx, dy = _letterbox(img)
    sess = get_session()
    pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]  # (5, N)

    conf = pred[4]
    keep = conf > CONF_THRES
    if not keep.any():
        return []
    cx, cy, bw, bh = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
    conf = conf[keep]

    # letterbox -> koordinat gambar asli
    x1 = (cx - bw / 2 - dx) / r
    y1 = (cy - bh / 2 - dy) / r
    boxes = np.stack([x1, y1, bw / r, bh / r], 1)  # xywh untuk NMSBoxes

    idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), CONF_THRES, IOU_THRES)
    if len(idx) == 0:
        return []

    h, w = img.shape[:2]
    out = []
    for i in np.array(idx).flatten():
        bx, by, bwi, bhi = boxes[i]
        out.append((
            max(0, int(bx)), max(0, int(by)),
            min(w, int(bx + bwi)), min(h, int(by + bhi)),
            float(conf[i]),
        ))
    out.sort(key=lambda b: -b[4])
    return out


def crop(img, box):
    """Potong plat dengan padding proporsional. None kalau kotak degenerate."""
    x1, y1, x2, y2, _ = box
    pw, ph = int((x2 - x1) * PAD_RATIO), int((y2 - y1) * PAD_RATIO)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
    if x2 - x1 < 16 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]
