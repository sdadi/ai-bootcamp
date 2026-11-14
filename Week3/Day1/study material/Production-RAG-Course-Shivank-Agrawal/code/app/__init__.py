"""Production-style RAG service (Module 13) by Shivank Agrawal - Future with Shivank.

    api.py      FastAPI: upload documents, check status, delete, ask questions
    worker.py   the background ingestion job (run by an RQ worker)
    db.py       Postgres + pgvector schema, connections and the hybrid (RRF) search query
    storage.py  where raw files live (local folder, or S3 / MinIO)
    reembed.py  switch to a new embedding model without downtime (blue/green)
"""
