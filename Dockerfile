# Base image: Python 3.12 versi slim (lebih kecil dari versi full)
FROM python:3.12-slim

# Set working directory di dalam container
WORKDIR /app

# Install system dependencies yang dibutuhkan beberapa Python package
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt dulu, lalu install (caching layer)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code aplikasi
COPY api/ ./api/
COPY models/ ./models/

# HF Spaces pakai port 7860
ENV PORT=7860
EXPOSE 7860

# Command yang dijalankan saat container start
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
