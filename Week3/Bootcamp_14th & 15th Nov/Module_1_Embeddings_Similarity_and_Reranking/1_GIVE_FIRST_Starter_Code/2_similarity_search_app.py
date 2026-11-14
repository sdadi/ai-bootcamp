"""
Exercise 1.2 — Similarity Search App (Top-K retrieval)
Created by Shivank Agarwal

What this app does
------------------
You paste some documents (one per line) and type a question.
The app turns every line and the question into embeddings (lists of numbers)
and shows the K documents whose meaning is closest to the question.

This is the "R" (Retrieval) in RAG.

Run it:
    python 2_similarity_search_app.py
Then open the link printed in the terminal (usually http://127.0.0.1:7860).
"""

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

# A small, free embedding model that runs on your laptop (no API key needed).
# It turns any sentence into a list of 384 numbers.
# The first run downloads it (about 90 MB).
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SAMPLE_DOCS = """The cat sat on the mat.
A dog ran in the park.
The stock market crashed today.
Bananas are a great source of potassium.
Apples and oranges are healthy fruits.
Sensex fell 800 points after the RBI announcement.
Virat Kohli scored a century in the final."""


# ----- TODO 1: Write the cosine similarity formula -----
# Hint: cosine(a, b) = (a · b) / (|a| × |b|)
# Hint: np.dot(a, b) and np.linalg.norm(a) are all you need
def cosine_similarity(a, b):
    return 0.0  # replace with the formula


def retrieve_top_k(documents_text, query, k):
    """Return the top-k documents as rows: [document, score]."""
    # 1. One document per non-empty line
    docs = [line.strip() for line in documents_text.strip().splitlines() if line.strip()]
    if not docs or not query.strip():
        return []

    # ----- TODO 2: Embed all documents and the query -----
    # Hint: model.encode(list_of_texts) returns one vector per text
    # Hint: Encode the query as a list with one item, then take [0]
    doc_embeddings = []      # replace with your code
    query_embedding = None   # replace with your code

    # ----- TODO 3: Score every document against the query and sort best-first -----
    # Hint: Build a list of (doc, score) pairs with zip(docs, doc_embeddings)
    # Hint: list.sort(key=lambda pair: pair[1], reverse=True) puts the highest score first
    scores = []  # replace with your code

    # 4. Keep only the first k results and format the score nicely for the table
    top = scores[: min(int(k), len(docs))]
    return [[doc, f"{score:.4f}"] for doc, score in top]


# ----------------------------------------------------------------------------
# The web page (Gradio builds it for us from Python)
# ----------------------------------------------------------------------------
with gr.Blocks(title="Similarity Search") as demo:
    gr.Markdown("# Similarity Search (Top-K)\nCreated by Shivank Agarwal")
    gr.Markdown("Enter documents (one per line), type a query, and get the K most similar documents.")

    with gr.Row():
        documents_input = gr.Textbox(label="Documents (one per line)", lines=12, value=SAMPLE_DOCS, scale=2)
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="Query", value="fruit and nutrition", lines=2)
            k_input = gr.Slider(label="K — number of results", value=3, minimum=1, maximum=20, step=1)
            retrieve_btn = gr.Button("Retrieve Top K", variant="primary")

    results_table = gr.Dataframe(headers=["Document", "Similarity Score"], label="Top-K Results", interactive=False, wrap=True)

    # When the button is clicked: call retrieve_top_k(documents, query, k) and show the result in the table
    retrieve_btn.click(fn=retrieve_top_k, inputs=[documents_input, query_input, k_input], outputs=results_table)

if __name__ == "__main__":
    demo.launch()
