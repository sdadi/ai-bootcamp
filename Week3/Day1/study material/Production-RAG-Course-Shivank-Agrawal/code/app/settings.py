"""Settings for the production app (environment variables / .env)."""
from __future__ import annotations

import os
from pathlib import Path

from rag_core.config import ROOT  # also loads .env

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "ingest")

# Raw files ("blobs"). Local folder by default; set S3_BUCKET to use S3 or MinIO instead.
BLOB_DIR = Path(os.getenv("BLOB_DIR", str(ROOT / "blobs")))
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")  # e.g. http://localhost:9000 for MinIO

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".htm", ".csv"}

CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "250"))
CANDIDATES = int(os.getenv("CANDIDATES", "40"))   # how many chunks each search returns before fusion
TOP_K = int(os.getenv("TOP_K", "5"))              # how many chunks go into the prompt
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
