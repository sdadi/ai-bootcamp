"""
LAB 11 - Security: PII, access control and prompt injection  (Module 11)
=======================================================================
What you will learn
  * Redact personal data BEFORE it is chunked, embedded, logged or sent to an LLM.
  * Enforce access control INSIDE retrieval (metadata filter), so forbidden text never reaches the prompt.
  * Indirect prompt injection: a document that tries to give orders to your LLM.
    Defences: detect + quarantine at ingestion, "sources are data" prompt, output checks.

How to run
    python labs/lab11_security.py

What to look for
  * The support tickets before and after redaction (Aadhaar is caught only if its checksum is valid).
  * The same salary question asked by a customer, an employee and an HR user.
  * With scanning OFF, the poisoned community tip shows up in the context for a coupon question.
"""
import _common
from rag_core.loaders import load_corpus
from rag_core.pipeline import RAGPipeline
from rag_core.security import detect_injection, output_leaks, redact_pii, verhoeff_ok
from rag_core.utils import banner, show_hits, step, table

embedder = _common.embedder_or_exit()
llm = _common.llm_status()

banner("1. PII redaction at ingestion")
tickets = next(d for d in load_corpus() if d.doc_id == "support_tickets_sample.txt")
clean, findings = redact_pii(tickets.text)
print("BEFORE:\n", tickets.text[200:620])
print("\nAFTER:\n", clean[200:620])
table(findings)
print("\nChecksum matters: '2345 6789 0124' valid Aadhaar format?", verhoeff_ok("234567890124"),
      "| a random 12-digit order id '2345 6789 0125'?", verhoeff_ok("234567890125"))
print("Placeholders like [PII_PHONE] are chosen so they do not create fake keyword matches for 'phone'.")

banner("2. Access control inside retrieval")
p = RAGPipeline(embedder=embedder, llm=llm, reranker=None).build()
q = "What is the salary band for an L5 Staff Engineer?"
for who, groups in [("customer", ["public"]), ("employee", ["public", "employees"]),
                    ("HR partner", ["public", "employees", "hr"])]:
    step(f"{who} asks: {q}")
    hits = p.retrieve(q, groups)
    print("   documents retrieved:", sorted({h.chunk.doc_id for h in hits}))
    print("   answer:", p.ask(q, groups).text[:160])
print("""
The confidential file never reaches the customer's or employee's prompt, so the LLM cannot leak it.
Never rely on a prompt like "do not reveal salaries" - a prompt is not a security boundary.""")

banner("3. Indirect prompt injection")
poison = next(d for d in load_corpus() if d.doc_id == "community_tips_untrusted.md")
print("Suspicious patterns found in community_tips_untrusted.md:", detect_injection(poison.text))
print("Ingestion report (quarantined chunks):", p.ingest_report["injection_flagged"])

q = "Are there any coupon codes or special return offers on electronics?"
unsafe = RAGPipeline(embedder=embedder, llm=llm, reranker=None, scan_injection=False).build()
for label, pipe in [("scanning OFF", unsafe), ("scanning ON (default)", p)]:
    step(label)
    hits = pipe.retrieve(q, ["public"])
    show_hits(hits, n=3)
    a = pipe.ask(q, ["public"])
    leaks = output_leaks(a.text, ["FREE100", "60 days"])
    print("   answer:", a.text[:200])
    print("   output check (banned strings found):", leaks or "none")
print("""
Defence in depth for RAG:
  1. Ingestion: detect + quarantine suspicious chunks; label sources by trust (trusted / untrusted).
  2. Prompt: sources are DATA; the system prompt says never follow instructions found inside them.
  3. Least privilege: the RAG bot has no tools that can send email, refund money or change data.
  4. Output: check the answer for things it must never contain (codes, URLs, secrets).
  5. Monitor: log and review flagged documents and refusals (OWASP LLM01 Prompt Injection,
     LLM08 Vector and Embedding Weaknesses).""")
print("\nNext: python labs/lab12_advanced_patterns.py")
