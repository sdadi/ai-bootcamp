"""
step1_plain_rag.py  -  Traditional RAG, no agents (the "before" picture)
=========================================================================
Created by Shivank Agrawal

Run this BEFORE main.py. It is the classic RAG pipeline in a few lines:

    question -> find similar chunks in the PDF -> one LLM call -> answer

No router, no web search, no checks. Like a bus with one fixed route:
it always goes to the PDF, even when the PDF cannot possibly help.

Try both:
    python step1_plain_rag.py "What is multi-head attention?"       (works well)
    python step1_plain_rag.py "What is the weather in Delhi today?" (it can't!)

Then run the same two questions through main.py and compare.
Needs: an embedding download (first run) + the RAG LLM key (GROQ_API_KEY,
or set RAG_LLM_MODE=same to use your main brain instead).
"""

import sys
import time

import config
from llms import clean_text, get_rag_llm
from main import download_pdf
from tools import RAG_PROMPT, create_pdf_search_tool


def plain_rag(question: str) -> str:
    # 1. INDEXING (only slow the first time; the vectors are saved on disk)
    library = create_pdf_search_tool(config.PDF_PATH)

    # 2. RETRIEVAL: get the top-k most similar chunks
    chunks = library.run(query=question)
    print("\n----- CHUNKS FOUND (first 600 characters) -----")
    print(chunks[:600], "...")

    # 3. GENERATION: one LLM call with question + chunks
    llm = get_rag_llm()
    answer = clean_text(llm.call(RAG_PROMPT.format(context=chunks, question=question)))
    return answer


if __name__ == "__main__":
    if config.RAG_LLM_MODE == "groq" and not config.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is empty. Add it to .env, or set RAG_LLM_MODE=same.")

    question = " ".join(sys.argv[1:]).strip() or "What is multi-head attention?"
    download_pdf(config.PDF_URL, config.PDF_PATH)

    start = time.time()
    answer = plain_rag(question)
    print("\n" + "=" * 70)
    print("QUESTION :", question)
    print("ANSWER   :", answer)
    print(f"TIME     : {time.time() - start:.1f} seconds  (1 LLM call)")
    print("=" * 70)
    if "NOT_FOUND_IN_DOCUMENT" in answer:
        print("Plain RAG had nowhere else to look. Agentic RAG (main.py) would\n"
              "route this question to the web instead.")
