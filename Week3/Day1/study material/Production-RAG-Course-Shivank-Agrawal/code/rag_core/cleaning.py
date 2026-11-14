"""
Chapter 5 - Cleaning extracted text.

Extraction gives you text that is "technically correct" but full of junk that hurts
chunking and embeddings. Each function below fixes one kind of junk and is safe to
run on any document.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .documents import Page


def normalize_unicode(text: str) -> str:
    """NFKC normalisation turns look-alike characters into their plain form.

    Examples: the ligature 'ﬂ' -> 'fl', full-width digits '５' -> '5',
    non-breaking spaces -> normal spaces. Without this, 'ﬂights' and 'flights'
    are different words for BM25 and look different to the embedding model.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = text.replace("​", "")  # zero-width space
    text = text.replace("<br>", " ")  # line breaks inside extracted table cells
    return text


def normalize_currency(text: str) -> str:
    """Write Indian rupee amounts one way: 'Rs. 3,000' / 'INR 3,000' -> '₹3,000'."""
    return re.sub(r"\b(?:Rs\.?|INR)\s?(?=\d)", "₹", text)


def fix_hyphenation(text: str) -> str:
    """Join words broken across lines: 'reimburse-\\nment' -> 'reimbursement'."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


_STRUCTURAL = re.compile(r"^\s*(#{1,6}\s|[-*•]\s|\d+[.)]\s|\|)")


def merge_broken_lines(text: str) -> str:
    """PDFs put a line break at the end of every visual line. Join lines that belong to
    the same paragraph, but keep headings, list items and table rows on their own lines."""
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if (out and out[-1] and not _STRUCTURAL.match(stripped) and not _STRUCTURAL.match(out[-1])
                and not out[-1].rstrip().endswith((".", ":", "?", "!"))):
            out[-1] = out[-1].rstrip() + " " + stripped
        else:
            out.append(stripped)
    return "\n".join(out)


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_signature(line: str) -> str:
    """'Page 3 of 7' and 'Page 4 of 7' get the same signature, so we can spot them."""
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_repeated_lines(pages: list[Page], min_share: float = 0.6) -> tuple[list[Page], list[str]]:
    """Remove headers/footers: short lines that repeat on most pages of a document."""
    if len(pages) < 2:
        return pages, []
    counts: Counter[str] = Counter()
    for p in pages:
        counts.update({_line_signature(l) for l in p.text.split("\n") if 0 < len(l.strip()) < 80})
    repeated = {sig for sig, c in counts.items() if c / len(pages) >= min_share}
    cleaned = []
    for p in pages:
        keep = [l for l in p.text.split("\n") if _line_signature(l) not in repeated]
        cleaned.append(Page(p.number, "\n".join(keep), p.method))
    return cleaned, sorted(repeated)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = normalize_currency(text)
    text = fix_hyphenation(text)
    text = merge_broken_lines(text)
    return collapse_whitespace(text)


def clean_pages(pages: list[Page]) -> tuple[list[Page], dict]:
    """Full cleaning pipeline for one document. Returns the cleaned pages and a small report."""
    before = sum(len(p.text) for p in pages)
    pages, removed = remove_repeated_lines(pages)
    pages = [Page(p.number, clean_text(p.text), p.method) for p in pages]
    after = sum(len(p.text) for p in pages)
    return pages, {"chars_before": before, "chars_after": after, "removed_repeated_lines": removed}
