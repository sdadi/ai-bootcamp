"""
Exercise 1.3 — Retrieve, then Rerank
Created by Shivank Agarwal

Two-step search, exactly like production RAG systems:

  Step 1  RETRIEVE (fast, rough)   -> bi-encoder: embed query and documents SEPARATELY,
                                      compare with cosine similarity, keep the top K.
  Step 2  RERANK   (slow, precise) -> cross-encoder: reads the query AND one document TOGETHER
                                      and gives a relevance score. Re-sort the K results.

Run it:
    python 3_reranking_app.py
Then open the link printed in the terminal (usually http://127.0.0.1:7860).
"""

import gradio as gr
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Bi-encoder: turns each text into a vector on its own (fast, good for searching many documents)
bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Cross-encoder: looks at (query, document) as a pair and scores how well the document answers the query.
# It was trained on MS MARCO, a large dataset of real search queries and answers.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

SAMPLE_DOCS = """COVID-19 has many symptoms.
COVID-19 symptoms are bad.
COVID-19 symptoms are not nice
COVID-19 symptoms are really bad.
COVID-19 is a disease caused by a virus. The most common symptoms are fever, chills, and sore throat, but there are a range of others.
COVID-19 symptoms can include: a high temperature or shivering (chills); a new, continuous cough; a loss or change to your sense of smell or taste; and many more
Dementia has the following symptom: Experiencing memory loss, poor judgment, and confusion.
COVID-19 has the following symptom: Experiencing memory loss, poor judgment, and confusion."""


# ----------------------------------------------------------------------------
# Core search logic (no UI code here)
# ----------------------------------------------------------------------------
def get_top_k(query: str, documents: list[str], k: int) -> list[tuple[str, float]]:
    """Step 1: bi-encoder retrieval. Returns [(document, cosine_score), ...] best first."""
    # ----- TODO 1: Embed the documents and the query, score them with cosine similarity, sort, keep top k -----
    # Hint: bi_encoder.encode(documents) -> matrix (one row per document)
    # Hint: bi_encoder.encode([query]) -> matrix with ONE row
    # Hint: cosine_similarity(query_matrix, doc_matrix)[0] -> one score per document
    return []  # replace with your code


def rerank(query: str, documents: list[str]) -> list[tuple[str, float]]:
    """Step 2: cross-encoder reranking. Returns [(document, relevance_score), ...] best first."""
    # ----- TODO 2: Score every (query, document) pair with the cross-encoder and sort best first -----
    # Hint: Build pairs like [(query, doc1), (query, doc2), ...]
    # Hint: cross_encoder.predict(pairs) -> one score per pair (higher = more relevant)
    return []  # replace with your code


# ----------------------------------------------------------------------------
# Button handlers (connect the UI to the logic above)
# ----------------------------------------------------------------------------
def to_table(results):
    return pd.DataFrame([(doc, round(float(score), 4)) for doc, score in results], columns=["Document", "Score"])


def on_retrieve(documents_text, query, k):
    docs = [line.strip() for line in documents_text.splitlines() if line.strip()]
    if not docs or not query.strip():
        raise gr.Error("Please enter some documents and a query.")
    return to_table(get_top_k(query.strip(), docs, int(k)))


def on_rerank(query, top_k_table):
    # We rerank exactly the documents shown in the Top-K table
    docs = [] if top_k_table is None else [str(d) for d in top_k_table["Document"].tolist() if str(d).strip()]
    if not docs:
        raise gr.Error("Click 'Retrieve Top K' first.")
    return to_table(rerank(query.strip(), docs))


# ----------------------------------------------------------------------------
# The web page
# ----------------------------------------------------------------------------
with gr.Blocks(title="Retrieve and Rerank") as demo:
    gr.Markdown("# Retrieve, then Rerank\nCreated by Shivank Agarwal")

    with gr.Row():
        documents_input = gr.Textbox(label="Documents (one per line)", lines=10, value=SAMPLE_DOCS, scale=2)
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="Query", value="COVID-19 Symptoms")
            k_input = gr.Number(label="K (number of results)", value=8, precision=0)

    retrieve_btn = gr.Button("1) Retrieve Top K", variant="primary")
    top_k_table = gr.Dataframe(label="Top-K Results (bi-encoder, cosine similarity)", headers=["Document", "Score"], interactive=False, wrap=True)

    rerank_btn = gr.Button("2) Rerank", variant="primary")
    reranked_table = gr.Dataframe(label="Reranked Results (cross-encoder)", headers=["Document", "Score"], interactive=False, wrap=True)

    retrieve_btn.click(fn=on_retrieve, inputs=[documents_input, query_input, k_input], outputs=top_k_table)
    rerank_btn.click(fn=on_rerank, inputs=[query_input, top_k_table], outputs=reranked_table)

if __name__ == "__main__":
    demo.launch()
