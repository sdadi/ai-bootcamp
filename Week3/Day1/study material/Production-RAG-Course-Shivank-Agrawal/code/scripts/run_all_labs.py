"""Run every lab one after another and report which ones passed (a smoke test for your setup).

    python scripts/run_all_labs.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
labs = sorted((ROOT / "labs").glob("lab*.py"))
results = []
for lab in labs:
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(lab)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    ok = proc.returncode == 0
    results.append((lab.name, ok, time.time() - t0))
    print(f"{'PASS' if ok else 'FAIL'}  {lab.name:40s} {time.time() - t0:6.1f}s")
    if not ok:
        print("   last lines of the error:\n   " + "\n   ".join(proc.stderr.strip().splitlines()[-6:]))
print(f"\n{sum(r[1] for r in results)}/{len(results)} labs passed")
sys.exit(0 if all(r[1] for r in results) else 1)
