"""
tools.py  -  The two "hands" of our agents
==========================================
Created by Shivank Agrawal

An agent (the brain) cannot open a PDF or browse the internet by itself.
It needs TOOLS - small Python functions it is allowed to call.
Like a doctor who needs a stethoscope and an X-ray machine to actually
examine a patient, the agent needs tools to actually fetch information.

We build two tools:

  1. rag_tool         -> searches OUR private PDF (the "library")
  2. web_search_tool  -> searches the INTERNET (the "Google trip")

The Router agent decides which one should be used for each question.

How does the agent know what a tool does?
  It reads the tool's NAME and DOCSTRING (the text in triple quotes).
  So a clear docstring is not decoration - it is an instruction to the AI.
"""

import re
from pathlib import Path

from crewai.tools import tool
from crewai_tools import PDFSearchTool
from tavily import TavilyClient

import config
from llms import clean_text, get_rag_llm

# ===========================================================================
# TOOL 1: THE RAG TOOL  (search inside our PDF)
# ===========================================================================
#
# What happens the FIRST time we create it (INDEXING - done once):
#   PDF  ->  split into small chunks  ->  each chunk turned into a vector
#        ->  vectors saved in a vector database (ChromaDB, on your laptop)
#
# What happens EVERY time the tool is called (RETRIEVAL + GENERATION):
#   question -> turned into a vector -> find the closest chunks
#            -> the fast RAG LLM reads those chunks and writes a short answer
# ===========================================================================


def create_pdf_search_tool(pdf_path: str) -> PDFSearchTool:
    """Build the 'library': load the PDF, chunk it, embed it, store it."""
    # One "shelf" (collection) per PDF + embedding model, so switching to a
    # different PDF or model never mixes old vectors with new ones.
    shelf = f"{Path(pdf_path).stem}_{config.EMBEDDING_MODEL.split('/')[-1]}"
    shelf = re.sub(r"[^a-zA-Z0-9_-]", "_", shelf)[:60]

    return PDFSearchTool(
        pdf=pdf_path,
        collection_name=shelf,
        limit=config.TOP_K,  # bring back the top-k most similar chunks
        config={
            # The embedding model: turns text into numbers (vectors).
            # It runs locally and is downloaded once from Hugging Face.
            "embedding_model": {
                "provider": "sentence-transformer",
                "config": {"model_name": config.EMBEDDING_MODEL},
            },
            # The vector database where the vectors are stored.
            "vectordb": {"provider": "chromadb", "config": {}},
        },
    )


RAG_PROMPT = """You are a careful assistant answering questions from a document.

Use ONLY the context below. Do not use outside knowledge.
If the context does not contain the answer, reply with exactly: NOT_FOUND_IN_DOCUMENT

Context from the document:
-----
{context}
-----

Question: {question}

Answer in 3 to 6 clear sentences:"""


def create_rag_tool(pdf_path: str):
    """Wrap the PDF search + the fast RAG LLM into ONE tool the agents can call."""
    pdf_search = create_pdf_search_tool(pdf_path)  # indexing happens here
    rag_llm = get_rag_llm()

    @tool("rag_tool")
    def rag_tool(question: str) -> str:
        """Search the private PDF knowledge base and return an answer grounded
        in the document, followed by the exact text chunks that were used.
        Input must be the user's full question as plain text."""
        # Step A - RETRIEVAL: find the most similar chunks in the vector DB
        chunks = pdf_search.run(query=question)

        if not chunks or "No relevant content found" in chunks:
            return "NOT_FOUND_IN_DOCUMENT"

        # Step B - GENERATION: the fast LLM writes an answer from those chunks
        answer = clean_text(rag_llm.call(RAG_PROMPT.format(context=chunks, question=question)))

        if "NOT_FOUND_IN_DOCUMENT" in answer:
            return "NOT_FOUND_IN_DOCUMENT"
        return f"ANSWER FROM DOCUMENT:\n{answer}\n\nCHUNKS USED:\n{chunks}"

    return rag_tool


# ===========================================================================
# TOOL 2: THE WEB SEARCH TOOL  (search the internet with Tavily)
# ===========================================================================
#
# The @tool decorator turns a normal Python function into an agent tool.
# Tavily is a search engine built for AI agents: it returns clean text
# instead of a messy web page full of ads.
# ===========================================================================


@tool("web_search_tool")
def web_search_tool(question: str) -> str:
    """Search the internet for fresh or general information that is NOT in the
    private PDF. Input must be the user's full question as plain text.
    Returns the top results with their title, link and content."""
    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(query=question, max_results=config.WEB_RESULTS)
    except Exception as error:  # no internet, wrong key, limit reached ...
        # Return a readable message instead of crashing the whole crew.
        return f"WEB_SEARCH_FAILED: {error}"

    results = response.get("results", [])
    if not results:
        return "NO_WEB_RESULTS"

    lines = []
    for number, item in enumerate(results, start=1):
        lines.append(
            f"[{number}] {item.get('title', '')}\n"
            f"Link: {item.get('url', '')}\n"
            f"{item.get('content', '')}\n"
        )
    return "\n".join(lines)
