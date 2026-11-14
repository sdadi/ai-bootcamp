"""
Chapter 5 - Getting text out of files.

One function per format, plus `load_corpus()` which loads every file in data/raw.

The most interesting part is `extract_pdf()`, which implements the PAGE and BLOCK
ROUTING idea from the course:

    for each page:
        if the page has (almost) no text but has images   -> it is a scan: OCR the page
        else:
            tables      -> extract as Markdown tables (keeps rows and columns)
            text blocks -> take the digital text directly (fast, exact)
            big images  -> OCR only that image region (cheap: no OCR on text we already have)

OCR uses Tesseract (free, local). If Tesseract is not installed we print a warning and
skip OCR, so the rest of the pipeline still runs.
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import warnings
from html.parser import HTMLParser
from pathlib import Path

from .documents import Document, Page

# ----------------------------------------------------------------------------- OCR

_TESSERACT_WARNED = False


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_image_bytes(png_bytes: bytes, psm: int = 3, lang: str = "eng") -> str:
    """Run Tesseract on a PNG image. Returns '' if Tesseract is missing."""
    global _TESSERACT_WARNED
    if not tesseract_available():
        if not _TESSERACT_WARNED:
            warnings.warn(
                "Tesseract is not installed, so scanned pages are skipped. "
                "Install it (see README: 'Install Tesseract') to enable OCR.",
                stacklevel=2,
            )
            _TESSERACT_WARNED = True
        return ""
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes))
    return pytesseract.image_to_string(img, lang=lang, config=f"--psm {psm}")


def _ocr_page_region(page, rect=None, dpi: int = 200, psm: int = 3) -> str:
    import pymupdf

    pix = page.get_pixmap(dpi=dpi, clip=rect, colorspace=pymupdf.csGRAY)
    return ocr_image_bytes(pix.tobytes("png"), psm=psm)


# ----------------------------------------------------------------------------- PDF


def _text_blocks_with_headings(page, skip_rects) -> list[tuple[float, str]]:
    """Read text blocks with their font sizes. A short block whose font is clearly bigger
    than the body text is a heading, so we mark it with Markdown '#'.

    This is a tiny example of LAYOUT-AWARE extraction: we use how the text looks
    (font size), not only what it says, to recover the document structure.
    """
    import statistics

    import pymupdf

    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
    sizes = [s["size"] for b in blocks for l in b["lines"] for s in l["spans"] if s["text"].strip()]
    if not sizes:
        return []
    body = statistics.median(sizes)
    out: list[tuple[float, str]] = []
    for b in blocks:
        rect = pymupdf.Rect(b["bbox"])
        if any(rect.intersects(r) for r in skip_rects):
            continue  # this text is already inside a Markdown table
        text = "\n".join("".join(s["text"] for s in l["spans"]) for l in b["lines"]).strip()
        if not text:
            continue
        biggest = max(s["size"] for l in b["lines"] for s in l["spans"])
        if biggest >= body * 1.15 and len(text) < 80 and "\n" not in text:
            level = "#" if biggest >= body * 1.45 else "##"
            text = f"{level} {text}"
        out.append((b["bbox"][1], text))
    return out


def extract_pdf(path: Path, ocr: bool = True, min_chars: int = 50, min_image_share: float = 0.05,
                dpi: int = 200) -> list[Page]:
    """Extract a PDF page by page, routing each page (and each image block) to the right method.

    min_chars        A page with fewer characters than this AND an image is treated as scanned.
    min_image_share  Images smaller than this share of the page (logos, icons) are ignored.
    """
    import pymupdf

    pages: list[Page] = []
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            images = page.get_image_info()
            page_area = page.rect.width * page.rect.height

            # ---- Route 1: a scanned page -> OCR the whole page
            if len(text.strip()) < min_chars and images:
                ocr_text = _ocr_page_region(page, dpi=dpi) if ocr else ""
                pages.append(Page(number, ocr_text.strip(), method="ocr"))
                continue

            # ---- Route 2: a digital page -> tables + text blocks (+ OCR for big images)
            parts: list[tuple[float, str]] = []  # (y position on page, text) so we keep reading order
            tables = page.find_tables().tables
            table_rects = [pymupdf.Rect(t.bbox) for t in tables]
            for t in tables:
                parts.append((t.bbox[1], t.to_markdown().strip()))

            for y0, block_text in _text_blocks_with_headings(page, table_rects):
                parts.append((y0, block_text))

            method = "digital" if not tables else "digital+table"
            for info in images:
                rect = pymupdf.Rect(info["bbox"])
                if rect.width * rect.height < min_image_share * page_area:
                    continue  # skip logos and icons
                if ocr:
                    ocr_text = _ocr_page_region(page, rect=rect, dpi=dpi, psm=6).strip()
                    if ocr_text:
                        parts.append((rect.y0, ocr_text))
                        method = "mixed"

            parts.sort(key=lambda p: p[0])  # simple top-to-bottom order (fine for 1-column pages)
            pages.append(Page(number, "\n\n".join(t for _, t in parts if t), method=method))
    return pages


# ----------------------------------------------------------------------------- HTML


class _HTMLToMarkdown(HTMLParser):
    """Turn HTML into Markdown-ish text: keep headings, paragraphs, lists and tables;
    drop scripts, styles, navigation and footers (boilerplate that pollutes chunks)."""

    SKIP = {"script", "style", "nav", "footer", "noscript", "svg", "form", "button", "head"}
    BLOCK = {"p", "div", "section", "article", "main", "header", "br", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.buf: list[str] = []
        self.skip_depth = 0
        self.prefix = ""
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def _flush(self) -> None:
        text = " ".join("".join(self.buf).split())
        if text:
            self.lines.append(self.prefix + text)
        self.buf, self.prefix = [], ""

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self.prefix = "#" * int(tag[1]) + " "
        elif tag == "li":
            self._flush()
            self.prefix = "- "
        elif tag == "table":
            self._flush()
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag in self.BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            if self.row:
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.lines.append(_rows_to_markdown(self.table))
            self.table = None
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6", "li", "p"} or tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.table is None:
            self.buf.append(data)

    def markdown(self) -> str:
        self._flush()
        return "\n\n".join(self.lines)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def extract_html(path: Path) -> list[Page]:
    parser = _HTMLToMarkdown()
    parser.feed(path.read_text(encoding="utf-8"))
    return [Page(1, parser.markdown(), method="text")]


# ----------------------------------------------------------------------------- CSV


def extract_csv(path: Path, title: str | None = None) -> list[Page]:
    """Turn each table row into a small, self-describing section.

    A bare row like "NM-PHN-4821,Nimbu Volt 5G,...,18999,12,7" means nothing to an
    embedding model. Writing every value next to its column name ("price_inr: 18999")
    makes each row searchable on its own, and one row becomes one chunk.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = [f"# {title or path.stem.replace('_', ' ').title()}",
             "Each section below is one row of the table, written out as text."]
    for row in rows:
        values = list(row.values())
        heading = " - ".join(v for v in values[:2] if v)
        lines.append(f"## {heading}")
        lines.extend(f"- {k}: {v}" for k, v in row.items())
    return [Page(1, "\n\n".join(lines[:2]) + "\n\n" + "\n".join(lines[2:]), method="table")]


# ----------------------------------------------------------------------------- dispatcher


def extract_file(path: Path, ocr: bool = True, title: str | None = None) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, ocr=ocr)
    if suffix in {".html", ".htm"}:
        return extract_html(path)
    if suffix == ".csv":
        return extract_csv(path, title=title)
    if suffix in {".md", ".txt"}:
        return [Page(1, path.read_text(encoding="utf-8"), method="text")]
    raise ValueError(f"Unsupported file type: {path.name}")


def load_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "_metadata.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("_about", None)
    return meta


def load_document(path: Path, metadata: dict | None = None, ocr: bool = True, clean: bool = True) -> Document:
    """Extract (and by default clean) one file into a Document."""
    from .cleaning import clean_pages

    metadata = dict(metadata or {})
    metadata.setdefault("title", path.stem.replace("_", " ").title())
    metadata["source_file"] = path.name
    pages = extract_file(path, ocr=ocr, title=metadata.get("title"))
    if clean:
        pages, report = clean_pages(pages)
        metadata["cleaning_report"] = report
    metadata["extraction_methods"] = sorted({p.method for p in pages})
    metadata["num_pages"] = len(pages)
    text = "\n\n".join(p.text for p in pages if p.text)
    return Document(doc_id=path.name, text=text, metadata=metadata, pages=pages)


def load_corpus(data_dir: Path | None = None, cache_dir: Path | None = None, ocr: bool = True,
                clean: bool = True, use_cache: bool = True, verbose: bool = False) -> list[Document]:
    """Load every document in data/raw (with metadata from _metadata.json).

    Extraction (especially OCR) is the slowest step, so results are cached as JSON in
    data/processed/. The cache is reused only if the source file has not changed.
    This is the same idea as production: extract once, index many times.
    """
    from .config import get_settings

    s = get_settings()
    data_dir = data_dir or s.data_dir
    cache_dir = cache_dir or s.processed_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_meta = load_metadata(data_dir)
    docs: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        if path.name.startswith(("_", ".")) or path.is_dir():
            continue
        cache_file = cache_dir / f"{path.name}.json"
        stamp = f"{path.stat().st_mtime_ns}-{path.stat().st_size}-ocr{int(ocr)}-clean{int(clean)}"
        if use_cache and cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("stamp") == stamp:
                pages = [Page(**p) for p in cached["pages"]]
                docs.append(Document(cached["doc_id"], cached["text"], cached["metadata"], pages))
                continue
        if verbose:
            print(f"  extracting {path.name} ...")
        doc = load_document(path, all_meta.get(path.name, {}), ocr=ocr, clean=clean)
        cache_file.write_text(json.dumps({
            "stamp": stamp, "doc_id": doc.doc_id, "text": doc.text, "metadata": doc.metadata,
            "pages": [p.__dict__ for p in doc.pages],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        docs.append(doc)
    return docs


def strip_markdown(text: str) -> str:
    """Remove Markdown symbols for display (not used for indexing)."""
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    return text.replace("**", "")
