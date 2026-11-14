"""
run_agent.py — the CAPSTONE: run the full deep-search agent from the command line.

Offline (default, no keys):
    python run_agent.py

Your own question about a REAL company (needs a live LLM + live search in .env):
    python run_agent.py "Is Acme Payments a reliable gateway partner today?" \
        --name "Acme Payments" --city Mumbai --country India \
        --industry "payment gateway" --domain acmepayments.com

Outputs land in runs/<timestamp>-deepsearch/: report.md, report.json, evidence.json, trace.jsonl

Please research COMPANIES and PRODUCTS, not private individuals. Profiling people raises
privacy and legal duties (India's DPDP Act, GDPR) — see Module 11 of the course.
"""

from __future__ import annotations

import argparse
import sys

from deepsearch.budget import Budget
from deepsearch.config import SETTINGS, describe
from deepsearch.orchestrator import DEFAULT_QUESTION, DeepSearchAgent, default_entity, report_to_markdown
from deepsearch.schemas import EntityAnchor


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Shivank's multi-agent deep search agent")
    p.add_argument("question", nargs="?", default=None, help="Research question (default: the NimbusPay case)")
    p.add_argument("--name", help="Company name, e.g. 'NimbusPay'")
    p.add_argument("--city", default="")
    p.add_argument("--country", default="")
    p.add_argument("--industry", default="", help="Comma-separated keywords, e.g. 'payment gateway,payments'")
    p.add_argument("--domain", default="", help="Official website domain, e.g. nimbuspay.example")
    p.add_argument("--exclude", default="", help="Comma-separated look-alike names or domains to ignore")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--max-usd", type=float, default=0.50, help="Hard cost limit for the run")
    p.add_argument("--max-tool-calls", type=int, default=120)
    p.add_argument("--no-memory", action="store_true", help="Do not read or write memory/lessons.json")
    p.add_argument("--sequential", action="store_true", help="Run workers one by one (to compare speed)")
    args = p.parse_args(argv)

    if args.name:
        entity = EntityAnchor(name=args.name, city=args.city, country=args.country, official_domain=args.domain,
                              industry_keywords=[x.strip() for x in args.industry.split(",") if x.strip()],
                              exclude=[x.strip() for x in args.exclude.split(",") if x.strip()])
        question = args.question or f"Is {args.name} a reliable partner today? Check leadership, licences, funding, " \
                                    f"reliability, security incidents, pricing and customer sentiment."
    else:
        entity, question = default_entity(), args.question or DEFAULT_QUESTION

    if SETTINGS.search_provider != "offline" and not args.name:
        print("Tip: with live search, describe a real company with --name/--city/--country/--industry/--domain.")
    print("Settings:", describe())
    agent = DeepSearchAgent(question, entity, budget=Budget(max_usd=args.max_usd, max_tool_calls=args.max_tool_calls),
                            max_iterations=args.max_iterations, parallel=not args.sequential,
                            use_memory=not args.no_memory)
    report = agent.run()
    print()
    print(report_to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
