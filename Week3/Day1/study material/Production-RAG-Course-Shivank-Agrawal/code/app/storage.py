"""
Blob storage for the ORIGINAL uploaded files.

Why store originals? Re-processing. When you improve extraction or chunking, or change the
embedding model, you re-run ingestion from the originals. The database only holds derived data.

Key layout: documents/<document_id>/<file name>
"""
from __future__ import annotations

from pathlib import Path

from . import settings


class LocalBlobStore:
    def __init__(self, root: Path = settings.BLOB_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def delete(self, key: str) -> None:
        path = self.root / key
        if path.exists():
            path.unlink()


class S3BlobStore:
    """Works with AWS S3 and with S3-compatible stores such as MinIO (pip install boto3)."""

    def __init__(self, bucket: str = settings.S3_BUCKET, endpoint_url: str = settings.S3_ENDPOINT_URL):
        import boto3

        self.bucket = bucket
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url or None)

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)


def get_store():
    return S3BlobStore() if settings.S3_BUCKET else LocalBlobStore()
