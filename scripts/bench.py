"""Self-check akurasi pipeline plat. Jalankan dari root project: python scripts/bench.py [dir]

Ground truth dibaca manual dari foto. Bukan framework, cuma skor + assert.
testdata/  = foto sulit (malam, blur, plat kecil)
testdata2/ = foto plat jelas, mirip cara petugas memotret
"""

import glob
import os
import sys
import time

import cv2

# Pastikan root project ada di path saat dijalankan dari subfolder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.pipeline as pipeline  # noqa: E402

TRUTH = {
    # testdata/ - NotIdeal
    "D.jpg": "D 5299 UCD",
    "WhatsApp Image 2021-10-08 at 18.35.06 (2).jpeg": "D 5299 UCD",
    "WhatsApp Image 2021-10-08 at 18.35.06.jpeg": "D 5299 UCD",
    "WhatsApp Image 2021-10-08 at 18.35.07 (1).jpeg": "D 5299 UCD",
    "WhatsApp Image 2021-10-08 at 18.35.07 (2).jpeg": "D 5299 UCD",
    "WhatsApp Image 2021-10-08 at 18.35.07.jpeg": "D 5299 UCD",
    "WhatsApp Image 2021-10-08 at 18.35.08.jpeg": "D 5299 UCD",
    "WhatsApp Image 2021-10-08 at 18.35.10 (1).jpeg": "D 1751 UQ",
    "WhatsApp Image 2021-10-08 at 18.35.10.jpeg": "D 1751 UQ",
    "dudukan-plat-nomor-mobil-2-dc3c.jpg": "B 1002 SJT",
    # testdata2/ - Ideal_Good
    "AA.jpg": "AA 4337 W",
    "AG.jpg": "AG 3738 IU",
    "B.jpg": "B 1991 JO",
    "BA.jpg": "BA 1632 BQ",
    "BP.jpg": "BP 1478 HD",
    "D2.jpg": "D 1512 XW",
    "DK.jpg": "DK 1356 CR",
    "H.jpg": "H 3420 BKE",
    "L.jpg": "L 4238 JS",
    "N(2).jpg": "N 3871 BAA",
    "N.jpg": "N 2776 HZ",
    "PB.jpg": "PB 212 GO",
    "S.jpg": "S 1403 TA",
    "W.jpg": "W 943 RD",
    "plat-nomor.jpg": "B 313 EEK",
}

# Default testdata di scripts/testdata dan scripts/testdata2
_SCRIPTS_DIR = os.path.dirname(__file__)
_DEFAULT_DIRS = [
    os.path.join(_SCRIPTS_DIR, "testdata"),
    os.path.join(_SCRIPTS_DIR, "testdata2"),
]


def run(folder):
    files = sorted(f for f in glob.glob(f"{folder}/*") if os.path.basename(f) in TRUTH)
    hit = 0
    wrong = 0
    total_t = 0.0
    for f in files:
        name = os.path.basename(f)
        want = TRUTH[name]
        img = cv2.imread(f)
        t = time.time()
        got, _, _ = pipeline.read_plate(img)
        el = time.time() - t
        total_t += el
        if got == want:
            hit += 1
            mark = "OK "
        else:
            wrong += got is not None
            mark = "X  "
        print(f"{mark} {name[:34]:36} got={str(got):14} want={want:12} {el:.2f}s", flush=True)
    n = len(files)
    print(f"-> {folder}: benar {hit}/{n}, salah-baca {wrong}, "
          f"rata-rata {total_t / n:.2f}s\n")
    return hit, n


if __name__ == "__main__":
    pipeline.get_reader()
    dirs = sys.argv[1:] or _DEFAULT_DIRS
    tot = sum(run(d)[0] for d in dirs)
    print("TOTAL BENAR:", tot)
