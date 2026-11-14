"""
LAB 00 - Check your setup  (5 minutes)
======================================
Run this first. It checks Python, the packages, Tesseract (for OCR), the embedding model,
the reranker, and your LLM key, and tells you how to fix anything that is missing.

    python labs/lab00_check_setup.py

Nothing here is required except Python + `pip install -r requirements.txt`.
Everything else has a fallback, and the lab output tells you what you will miss.
"""
import importlib
import platform
import shutil
import sys

import _common  # noqa: F401  (sets up imports)

OK, BAD, WARN = "[ok]  ", "[FIX] ", "[note]"


def check(label, ok, fix=""):
    print(f"{OK if ok else BAD}{label}" + ("" if ok else f"\n        -> {fix}"))
    return ok


print(f"Python {platform.python_version()} on {platform.system()}")
check("Python 3.10 or newer", sys.version_info >= (3, 10), "Install Python 3.11 or 3.12 from python.org")

for pkg in ["numpy", "fastembed", "pymupdf", "faiss", "dotenv", "openai", "anthropic"]:
    try:
        importlib.import_module(pkg)
        check(f"package {pkg}", True)
    except Exception:
        check(f"package {pkg}", False, "pip install -r requirements.txt   (inside your virtual environment)")

check("Tesseract OCR installed (needed for scanned PDFs in Lab 02)", shutil.which("tesseract") is not None,
      "macOS: brew install tesseract | Ubuntu: sudo apt install tesseract-ocr | "
      "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH. "
      "(Optional: without it, scanned pages are skipped.)")

print("\nLoading the embedding model (first time downloads ~70 MB) ...")
try:
    from rag_core.embeddings import get_embedder

    emb = get_embedder()
    v = emb.embed_query("hello")
    check(f"embedding model {emb.name} works, {emb.dim} dimensions", len(v) == emb.dim)
except Exception as e:
    check("embedding model", False, f"{e.__class__.__name__}: {str(e)[:120]}. If huggingface.co is blocked, "
          "set EMBEDDER=wordllama in .env (fully offline).")

print("\nLoading the reranker (first time downloads ~80 MB) ...")
from rag_core.rerank import get_reranker

rr = get_reranker()
check(f"reranker: {rr.name}", "fallback" not in rr.name,
      "The cross-encoder could not be downloaded; Lab 08 will use a weaker keyword fallback.")

print()
from rag_core.llm import LLM

llm = LLM()
if not llm.available:
    print(f"{WARN} No LLM configured. Labs still run (extractive answers).")
    print("        To use a real LLM, copy .env.example to .env and set OPENAI_API_KEY or ANTHROPIC_API_KEY,")
    print("        or install Ollama (https://ollama.com), run `ollama pull llama3.2`, and set LLM_PROVIDER=ollama.")
else:
    try:
        r = llm.generate("Reply with one word.", "Say: ready")
        check(f"LLM {llm.provider}/{llm.model} replied '{r.text[:30]}' in {r.latency_ms:.0f} ms", True)
    except Exception as e:
        check(f"LLM {llm.provider}/{llm.model}", False, f"{e.__class__.__name__}: {str(e)[:200]}  "
              "(check the key, the model name in LLM_MODEL, and your billing)")

print("\nFor the production app (Module 13) you also need Docker Desktop:",
      "found" if shutil.which("docker") else "not found (install from docker.com when you reach Module 13)")
print("\nDone. Next: python labs/lab01_rag_in_60_lines.py")
