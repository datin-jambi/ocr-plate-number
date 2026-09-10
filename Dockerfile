FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_PORT=5000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py detector.py ./
COPY models/ ./models/

EXPOSE 5000

# 1 worker: EasyOCR model is loaded per-process (RAM heavy). Scale with replicas.
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 -t 120 -b 0.0.0.0:${APP_PORT} app:app"]
