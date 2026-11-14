# Image for the API and the ingestion worker (Module 13)
FROM python:3.12-slim

# Tesseract for OCR of scanned PDFs
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_core ./rag_core
COPY app ./app
COPY eval ./eval
COPY data ./data

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
