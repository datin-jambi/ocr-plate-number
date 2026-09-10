# Plate Recognition API

Microservice standalone untuk deteksi & OCR plat nomor kendaraan Indonesia.
Flask + OpenCV (headless) + EasyOCR, siap deploy via Gunicorn atau Docker Compose.

---

## 1. Struktur Proyek

```
app.py                 Aplikasi Flask: route, config env, OCR singleton, regex parser
test_parser.py         Self-check regex parser plat nomor (tanpa framework test)
requirements.txt       Dependensi Python
.env / .env.example    Konfigurasi environment
Dockerfile             Image production (python:3.10-slim + libgl1)
docker-compose.yml     Orkestrasi container
.gitignore
```

---

## 2. Environment Variables

| Variabel          | Default   | Keterangan                                  |
|-------------------|-----------|---------------------------------------------|
| `APP_PORT`        | `5000`    | Port HTTP                                   |
| `APP_HOST`        | `0.0.0.0` | Host bind                                   |
| `FLASK_ENV`       | `development` | `development` = debug ON, `production` = OFF |
| `ALLOWED_ORIGINS` | `*`       | CORS. Pisahkan dengan koma untuk multi-origin |
| `OCR_GPU`         | `False`   | `True` untuk inference CUDA                 |

Setup:

```bash
cp .env.example .env
```

---

## 3. Setup Lokal (Virtual Environment)

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Catatan: instalasi pertama mengunduh PyTorch (dependensi EasyOCR, ~2 GB).
Saat request OCR pertama, EasyOCR juga mengunduh model detector/recognizer
ke `~/.EasyOCR`. Pastikan koneksi internet tersedia pada run pertama.

---

## 4. Menjalankan Server

Development (Flask dev server, auto-reload):

```bash
python app.py
```

Production (Gunicorn):

```bash
gunicorn -w 1 -k gthread --threads 4 -t 120 -b 0.0.0.0:${APP_PORT:-5000} app:app
```

Gunakan `-w 1`. Setiap worker memuat model EasyOCR sendiri di RAM
(~1 GB/worker); untuk skala tambah replika container, bukan worker.

---

## 5. Docker Compose

```bash
docker compose build
docker compose up -d
docker compose logs -f
docker compose down
```

Port di-expose sesuai `APP_PORT` di `.env`. Model EasyOCR di-cache pada
named volume `easyocr-models`, sehingga rebuild tidak mengunduh ulang.
Restart policy: `unless-stopped`.

---

## 6. Endpoint

### GET /health — liveness

```bash
curl http://localhost:5000/health
```

```json
{ "status": "ok", "service": "plate-recognition-api" }
```

### GET /health/ocr — readiness

Memicu load model EasyOCR ke memori. Request pertama bisa lambat
(unduh + inisialisasi model), berikutnya instan.

```bash
curl http://localhost:5000/health/ocr
```

```json
{ "status": "ready", "ocr_engine": "easyocr", "gpu_enabled": false }
```

### POST /api/detect-plate — deteksi plat

Payload `multipart/form-data`, field `file` (atau `image`).
Ekstensi yang diterima: `jpg`, `jpeg`, `png`, `webp`. Maksimal 10 MB.
Gambar dibaca langsung di memori, tidak pernah ditulis ke disk.

```bash
curl -X POST http://localhost:5000/api/detect-plate \
  -F "file=@/path/to/plat.jpg"
```

Sukses (HTTP 200):

```json
{ "status": "success", "plate_number": "B 1234 ABC" }
```

Tidak terdeteksi (HTTP 200):

```json
{ "status": "not_found", "message": "Plat nomor tidak terdeteksi" }
```

File kosong / ekstensi salah / gambar rusak (HTTP 400):

```json
{ "status": "error", "message": "Ekstensi tidak didukung. Gunakan: jpeg, jpg, png, webp" }
```

---

## 7. Testing Parser

```bash
python test_parser.py
```

Memvalidasi regex plat Indonesia (`B 1234 ABC`, `DK9999XYZ`, dst.)
tanpa perlu menjalankan OCR.

---

## 8. Pipeline Deteksi

1. Baca upload ke buffer memori, decode via `cv2.imdecode`.
2. Preprocessing: grayscale → bilateral filter → threshold Otsu.
3. EasyOCR `readtext` dengan allowlist `A-Z0-9`.
4. Gabungkan fragmen teks, cocokkan regex
   `([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})?`.
5. Kembalikan match pertama yang lengkap, format `HURUF ANGKA HURUF`.
