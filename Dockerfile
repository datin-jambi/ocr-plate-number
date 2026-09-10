FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_PORT=5000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Koneksi lambat: timeout longgar + retry, cache mount agar unduhan tidak hilang saat retry.
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

COPY requirements.txt .
# torch CPU-only (~200MB); wheel default PyPI adalah build CUDA (~2GB).
# Pin "+cpu" hanya ada di index PyTorch, PyPI tetap dipakai untuk dependensi torch.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install torch==2.4.1+cpu torchvision==0.19.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
    && pip install -r requirements.txt

COPY wsgi.py ./
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 5051

# 1 worker: EasyOCR model is loaded per-process (RAM heavy). Scale with replicas.
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 -t 120 -b 0.0.0.0:${APP_PORT} wsgi:app"]
