FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-online.txt /tmp/requirements-online.txt
RUN pip install --no-cache-dir -r /tmp/requirements-online.txt

COPY backend /app

RUN mkdir -p /app/generated/files /app/generated/jobs /app/app/evaluation/results

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
