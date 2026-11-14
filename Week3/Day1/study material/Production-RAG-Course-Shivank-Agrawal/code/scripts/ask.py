"""
Ask the running API a question as a given user group.

    python scripts/ask.py "What does error E102 mean?" --groups public,employees
    python scripts/ask.py "What is the salary band for L5?" --groups public          # should refuse
    python scripts/ask.py "What is the salary band for L5?" --groups public,employees,hr
"""
import argparse
import json

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--groups", default="public")
ap.add_argument("--api", default="http://localhost:8000")
ap.add_argument("--archived", action="store_true", help="also search archived documents")
a = ap.parse_args()
r = httpx.post(f"{a.api}/query", json={"question": a.question, "include_archived": a.archived},
               headers={"X-User-Groups": a.groups}, timeout=120)
r.raise_for_status()
print(json.dumps(r.json(), indent=2, ensure_ascii=False))
