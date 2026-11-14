"""
Upload every sample document to the running API, then wait until all are indexed.

    python scripts/ingest_samples.py                       # API at http://localhost:8000
    python scripts/ingest_samples.py --api http://host:8000

Run it twice: the second run shows idempotent uploads ("duplicate": true).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000")
args = ap.parse_args()

meta = json.loads((RAW / "_metadata.json").read_text(encoding="utf-8"))
ids = {}
with httpx.Client(base_url=args.api, timeout=60) as client:
    for path in sorted(RAW.iterdir()):
        if path.name.startswith(("_", ".")):
            continue
        m = meta.get(path.name, {})
        form = {"title": m.get("title", ""), "department": m.get("department", "general"),
                "access": ",".join(m.get("access", ["public"])), "doc_type": m.get("doc_type", "document"),
                "version": str(m.get("version", "1")), "status": m.get("status", "active"),
                "trust": m.get("trust", "trusted")}
        r = client.post("/documents", data=form, files={"file": (path.name, path.read_bytes())})
        r.raise_for_status()
        ids[path.name] = r.json()["id"]
        print(f"uploaded {path.name:36s} -> {r.json()}")

    print("\nWaiting for the worker to index everything (is `rq worker ingest --with-scheduler` running?)")
    deadline = time.time() + 600
    while time.time() < deadline:
        docs = {d["id"]: d for d in client.get("/documents").json()}
        states = {name: docs[i]["status"] for name, i in ids.items() if i in docs}
        pending = [n for n, s in states.items() if s not in {"indexed", "failed"}]
        print(f"  indexed {sum(s == 'indexed' for s in states.values())}/{len(states)}  "
              f"failed {sum(s == 'failed' for s in states.values())}  pending {len(pending)}")
        if not pending:
            break
        time.sleep(3)
    for name, i in ids.items():
        d = docs.get(i, {})
        if d.get("status") == "failed":
            print(f"  FAILED {name}: {d.get('error')}")
    sys.exit(0 if all(docs.get(i, {}).get("status") == "indexed" for i in ids.values()) else 1)
