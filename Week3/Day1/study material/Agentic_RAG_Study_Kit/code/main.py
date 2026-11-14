"""
main.py  -  Press START on the whole Agentic RAG system
=======================================================
Created by Shivank Agrawal

This file does not contain any "intelligence". Like a cricket captain who
doesn't bowl every over, it only sets the field and starts the game:
  1. makes sure the PDF is available
  2. builds the 2 tools
  3. hires the 5 agents
  4. gives them their 5 tasks
  5. puts everyone in a Crew and calls kickoff()

Run it:
    python main.py                                         (default question)
    python main.py "What is multi-head attention?"         (goes to the PDF)
    python main.py "What is the latest version of Python?" (goes to the web)
    python main.py --chat                                  (ask many questions)
"""

import os
import sys
import time
import urllib.request
from datetime import datetime

from crewai import Crew, Process

import config
from agents import (
    create_answer_generator_agent,
    create_answer_grader_agent,
    create_hallucination_grader_agent,
    create_retriever_agent,
    create_router_agent,
)
from llms import clean_text
from tasks import (
    create_answer_task,
    create_grader_task,
    create_hallucination_task,
    create_retriever_task,
    create_router_task,
)
from tools import create_rag_tool, web_search_tool

DEFAULT_QUESTION = "What is the self-attention mechanism in Transformers?"


def download_pdf(url: str, save_path: str) -> None:
    """Download the PDF only if we don't already have it."""
    if os.path.exists(save_path):
        print(f"PDF already present: {save_path}")
        return
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    print(f"Downloading PDF from {url} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(save_path, "wb") as f:
        f.write(response.read())
    print(f"Saved to {save_path}")


def build_crew() -> Crew:
    # ---- 1. TOOLS ---------------------------------------------------------
    rag_tool = create_rag_tool(config.PDF_PATH)  # builds the vector DB (indexing)

    # ---- 2. AGENTS --------------------------------------------------------
    router = create_router_agent()
    retriever = create_retriever_agent()
    grader = create_answer_grader_agent()
    hallucination_grader = create_hallucination_grader_agent()
    answer_generator = create_answer_generator_agent()

    # ---- 3. TASKS (each one can read the output of the ones before it) ----
    router_task = create_router_task(router)
    retriever_task = create_retriever_task(retriever, router_task, rag_tool, web_search_tool)
    grader_task = create_grader_task(grader, retriever_task)
    hallucination_task = create_hallucination_task(hallucination_grader, retriever_task, grader_task)
    answer_task = create_answer_task(
        answer_generator, retriever_task, grader_task, hallucination_task, web_search_tool
    )

    # ---- 4. CREW ----------------------------------------------------------
    # Process.sequential = tasks run one after another, in the list order.
    # max_rpm = at most this many LLM calls per minute (protects free tiers).
    return Crew(
        agents=[router, retriever, grader, hallucination_grader, answer_generator],
        tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
        process=Process.sequential,
        max_rpm=config.MAX_RPM,
        verbose=True,
    )


def one_word(task) -> str:
    """Read a task's short result (like 'vectorstore' or 'yes') safely."""
    raw = clean_text(task.output.raw if task.output else "?")
    return raw.split()[0].strip(".'\"").lower() if raw else "?"


def ask(crew: Crew, question: str) -> str:
    start = time.time()
    # kickoff() starts the team. `inputs` fills every {question} placeholder.
    result = crew.kickoff(inputs={"question": question})
    seconds = time.time() - start
    answer = clean_text(result)

    route, _, relevant, grounded, _ = (one_word(t) for t in crew.tasks)
    summary = f"route={route}  relevant={relevant}  grounded={grounded}  time={seconds:.0f}s"

    print("\n" + "=" * 70)
    print("QUESTION :", question)
    print("ANSWER   :", answer)
    print("JOURNEY  :", summary)
    print("=" * 70)

    # Keep a diary of every question, so you can compare runs later.
    with open(config.ANSWERS_LOG, "a", encoding="utf-8") as log:
        log.write(f"## {datetime.now():%Y-%m-%d %H:%M}  {question}\n\n{answer}\n\n`{summary}`\n\n")
    return answer


def main() -> None:
    problems = config.check_keys()
    if problems:
        print("\nPlease fix your .env file first:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    download_pdf(config.PDF_URL, config.PDF_PATH)
    crew = build_crew()

    if "--chat" in sys.argv:
        print("\nChat mode. Type a question and press Enter. Type 'exit' to stop.")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in ("exit", "quit", ""):
                break
            ask(crew, question)
        return

    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    ask(crew, question)


if __name__ == "__main__":
    main()
