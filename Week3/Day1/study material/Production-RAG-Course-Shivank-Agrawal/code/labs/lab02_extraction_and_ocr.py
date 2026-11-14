"""
LAB 02 - Text extraction, page routing, OCR and cleaning  (Module 5)
===================================================================
What you will learn
  * Why "just read the PDF" fails: scanned pages have NO text, images hide text, tables collapse.
  * Page/block ROUTING: digital text where it exists, OCR only where it is needed.
  * Tables -> Markdown, HTML -> clean text (no menus/footers/scripts), CSV rows -> searchable text.
  * Cleaning: ligatures (ﬂ), repeated headers/footers, currency formats.

How to run
    python labs/lab02_extraction_and_ocr.py
    python labs/lab02_extraction_and_ocr.py --vlm     # also try VLM OCR (needs an API key, costs a little)

What to look for
  * The route chosen for each PDF page: digital / digital+table / mixed / ocr.
  * Raw vs cleaned text of the travel policy: "ﬂights" -> "flights", "Page 1 of 3" removed.
  * The OCR text of the scanned warranty page.
"""
import sys

import pymupdf

import _common
from rag_core.cleaning import clean_pages, normalize_unicode
from rag_core.loaders import extract_csv, extract_html, extract_pdf, tesseract_available
from rag_core.utils import banner, step, table

RAW = _common.CODE_DIR / "data" / "raw"

banner("1. What a naive extractor sees")
for name in ["travel_reimbursement_policy.pdf", "warranty_terms_scanned.pdf", "vendor_onboarding_mixed.pdf"]:
    with pymupdf.open(RAW / name) as pdf:
        rows = [{"file": name, "page": i + 1, "chars of text": len(p.get_text()),
                 "images": len(p.get_images())} for i, p in enumerate(pdf)]
    table(rows)
print("\nThe scanned PDF has 0 characters of text: a naive RAG would silently index NOTHING from it.")

banner("2. Routing every page (and every big image) to the right method")
print("Tesseract OCR available:", tesseract_available())
for name in ["travel_reimbursement_policy.pdf", "warranty_terms_scanned.pdf", "vendor_onboarding_mixed.pdf"]:
    pages = extract_pdf(RAW / name)
    table([{"file": name, "page": p.number, "route": p.method, "chars": len(p.text),
            "starts with": p.text[:60]} for p in pages])

step("OCR output of the scanned warranty page")
print(extract_pdf(RAW / "warranty_terms_scanned.pdf")[0].text[:700] or "(empty: install Tesseract to enable OCR)")

step("The checklist that was only an IMAGE inside the mixed PDF (found by block-level OCR)")
mixed = extract_pdf(RAW / "vendor_onboarding_mixed.pdf")
print(mixed[-1].text[-400:])

banner("3. Tables stay tables")
travel = extract_pdf(RAW / "travel_reimbursement_policy.pdf")
md_table = [l for l in travel[0].text.split("\n") if l.startswith("|")]
print("\n".join(md_table))
with pymupdf.open(RAW / "travel_reimbursement_policy.pdf") as pdf:
    flat = pdf[0].get_text()
start = flat.find("City tier")
print("\nCompare: the same table as plain text (row/column meaning is lost):\n", " ".join(flat[start:start + 160].split()))

banner("4. Cleaning")
raw_text = travel[0].text
cleaned_pages, report = clean_pages(travel)
print("Removed repeated header/footer patterns:", report["removed_repeated_lines"])
print(f"Characters: {report['chars_before']} -> {report['chars_after']}")
for bad in ["ﬂ", "Rs. ", "Page 1 of 3"]:
    print(f"  '{bad}' in raw text: {bad in raw_text}   in cleaned text: {bad in cleaned_pages[0].text}")
print("  NFKC example:", repr("ﬂights ５ days"), "->", repr(normalize_unicode("ﬂights ５ days")))

banner("5. HTML and CSV")
html = extract_html(RAW / "nimbu_plus_membership.html")[0].text
print("HTML -> Markdown (menus, footer and scripts removed):\n", html[:500], "...")
print("\nCSV row -> self-describing text (one row = one future chunk):")
print("\n".join(extract_csv(RAW / "product_catalog.csv", "Product Catalog")[0].text.split("\n")[2:10]))

if "--vlm" in sys.argv:
    banner("6. VLM OCR (vision model reads the page image)")
    llm = _common.llm_status()
    if llm.available:
        with pymupdf.open(RAW / "warranty_terms_scanned.pdf") as pdf:
            png = pdf[0].get_pixmap(dpi=150).tobytes("png")
        r = llm.read_image(png)
        print(r.text[:800], f"\n\n[{r.latency_ms:.0f} ms, {r.input_tokens} input tokens - compare with Tesseract's cost: 0]")
    else:
        print("Needs an LLM API key with a vision-capable model.")

print("\nNext: python labs/lab03_chunking_strategies.py")
