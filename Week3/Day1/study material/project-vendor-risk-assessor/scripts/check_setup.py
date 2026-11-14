"""scripts/check_setup.py — run this first. Tells you what works and what to install."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print(f"Python {sys.version.split()[0]}  ({'OK' if sys.version_info >= (3, 10) else 'need 3.10+'})")

REQUIRED = {"pydantic": "core", "yaml": "requirements.yaml", "dotenv": ".env"}
OPTIONAL = {"langgraph": "graph engine (plain Python fallback works)", "openai": "live LLM", "ddgs": "live web search (free)",
            "tavily": "live web search (API)", "trafilatura": "clean page text", "fastapi": "app/api.py", "streamlit": "app/ui.py",
            "pytest": "tests"}
for mod, why in REQUIRED.items():
    ok = importlib.util.find_spec(mod) is not None
    print(f"  [{'OK' if ok else 'MISSING'}] {mod:<12} {why}")
for mod, why in OPTIONAL.items():
    ok = importlib.util.find_spec(mod) is not None
    print(f"  [{'OK' if ok else '--'}] {mod:<12} {why}  (optional)")

from assessor.config import SETTINGS, POLICY_DIR, VENDOR_CORPUS  # noqa: E402

print(f"\nLLM_PROVIDER={SETTINGS.llm_provider}  SEARCH_PROVIDER={SETTINGS.search_provider}  AS_OF={SETTINGS.as_of}")
print(f"policies: {len([p for p in POLICY_DIR.glob('*.md')])} files   vendor corpus: {'OK' if VENDOR_CORPUS.exists() else 'MISSING'}")
if SETTINGS.llm_provider != "mock":
    try:
        cfg = SETTINGS.resolved_llm()
        print(f"live LLM: {cfg['model']} @ {cfg['base_url']}  key: {'set' if cfg['api_key'] else 'MISSING'}")
    except Exception as exc:  # noqa: BLE001
        print(f"live LLM config problem: {exc}")

try:
    from assessor.policy_rag import PolicyLibrary
    lib = PolicyLibrary()
    print(f"policy library: {len(lib.chunks)} clause chunks indexed")
    print("\nAll good. Next:  python run_assessor.py")
except Exception as exc:  # noqa: BLE001
    print(f"\nProblem loading the policy library: {exc}")
