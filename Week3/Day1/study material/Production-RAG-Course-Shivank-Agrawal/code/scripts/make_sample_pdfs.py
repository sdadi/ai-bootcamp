"""
Generate the three sample PDFs used in Lab 02 (extraction and OCR).

You do NOT need to run this: the PDFs are already in data/raw/.
It is here so you can see how they were made, and regenerate them if you delete them.

    python scripts/make_sample_pdfs.py

What it creates (all content is fictional, for teaching):
  1. travel_reimbursement_policy.pdf  - a normal "digital" PDF with a real text layer,
     a table, and a repeated header/footer on every page (to practise cleaning).
  2. warranty_terms_scanned.pdf       - a "scanned" PDF: every page is only a picture,
     so plain text extraction returns nothing and we must use OCR.
  3. vendor_onboarding_mixed.pdf      - page 1 is digital text; page 2 has digital text
     AND an embedded image of a checklist (text inside the image needs OCR).

Author: Shivank Agrawal (Future with Shivank)
"""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf  # PyMuPDF
from PIL import Image, ImageFilter

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

A4 = pymupdf.paper_rect("a4")
MARGIN = 56  # points (about 2 cm)

CSS = """
body { font-family: sans-serif; font-size: 11pt; line-height: 1.35; }
h1 { font-size: 17pt; margin: 0 0 8pt 0; }
h2 { font-size: 13pt; margin: 12pt 0 4pt 0; }
table { border-collapse: collapse; margin: 6pt 0; }
th, td { border: 1px solid #000; padding: 4pt 8pt; text-align: left; }
th { background: #ddd; }
"""


def _page_with_html(doc: pymupdf.Document, html: str, header: str | None, footer: str | None):
    page = doc.new_page(width=A4.width, height=A4.height)
    body = pymupdf.Rect(MARGIN, MARGIN + 24, A4.width - MARGIN, A4.height - MARGIN - 24)
    page.insert_htmlbox(body, html, css=CSS)
    if header:
        page.insert_text((MARGIN, MARGIN), header, fontsize=8, color=(0.35, 0.35, 0.35))
    if footer:
        page.insert_text((MARGIN, A4.height - MARGIN + 10), footer, fontsize=8, color=(0.35, 0.35, 0.35))
    return page


def _save(doc: pymupdf.Document, out: Path) -> None:
    doc.subset_fonts()                      # keep only the glyphs we use
    doc.save(out, garbage=4, deflate=True)  # remove unused objects, compress streams


# ---------------------------------------------------------------- 1. digital PDF
TRAVEL_PAGES = [
    """
<h1>Travel and Expense Reimbursement Policy (FIN-007)</h1>
<p>This policy applies to all Nimbu Mart employees who travel for work. It explains what the
company pays for, the limits per city tier, and how to file a claim. Version 3, effective
1 July 2025. Owner: Finance team.</p>
<h2>1. Booking travel</h2>
<p>All flights must be booked in economy class through the Nimbu Travel desk at least
14 days before the travel date. Business class is not allowed for any level. Trains may be
booked in 2AC or 3AC. Cabs to and from the airport are reimbursed at actual cost.</p>
<h2>2. Daily allowance and hotel limits</h2>
<p>The daily allowance (per diem) covers meals and local travel. Hotel limits are per night,
including taxes. City tiers follow the same list used for customer delivery.</p>
<table>
<tr><th>City tier</th><th>Daily allowance</th><th>Hotel limit per night</th></tr>
<tr><td>Tier-1 metro</td><td>Rs. 3,000</td><td>Rs. 7,000</td></tr>
<tr><td>Tier-2 city</td><td>Rs. 2,000</td><td>Rs. 4,500</td></tr>
<tr><td>Tier-3 town</td><td>Rs. 1,500</td><td>Rs. 3,000</td></tr>
</table>
""",
    """
<h2>3. Filing a claim</h2>
<p>Submit every claim in the Nimbu Expense portal within 30 days of returning from the trip.
Claims filed after 30 days need approval from the Finance Controller. Attach a GST invoice
for every hotel stay and every flight. Bills without a GST invoice are reimbursed only up to
Rs. 500 per day.</p>
<h2>4. Approvals</h2>
<p>Your reporting manager approves the claim first. Finance then checks it against this
policy. Approved claims are paid in the next payroll cycle.</p>
<h2>5. Things we do not reimburse</h2>
<p>Alcohol, laundry for trips shorter than 4 nights, traffic fines, and upgrades paid to
airlines or hotels are not reimbursed.</p>
""",
    """
<h2>6. International travel</h2>
<p>International travel needs written approval from a Vice President before booking. The
daily allowance for international travel is USD 75 per day. Hotel limits for international
travel are decided case by case by the Finance Controller.</p>
<h2>7. Questions</h2>
<p>Write to the Finance helpdesk on the People portal. Expected response time is 2 working
days.</p>
""",
]


def make_travel_pdf() -> Path:
    doc = pymupdf.open()
    n = len(TRAVEL_PAGES)
    for i, html in enumerate(TRAVEL_PAGES, start=1):
        _page_with_html(
            doc,
            html,
            header="Nimbu Mart | Internal | Finance Policy FIN-007",
            footer=f"Page {i} of {n}",
        )
    doc.set_metadata({"title": "Travel and Expense Reimbursement Policy", "author": "Nimbu Mart Finance"})
    out = RAW / "travel_reimbursement_policy.pdf"
    _save(doc, out)
    return out


# ---------------------------------------------------------------- 2. scanned PDF
WARRANTY_HTML = """
<h1>Nimbu Care Warranty Terms</h1>
<p>Every product sold by Nimbu Mart carries the manufacturer warranty listed below.
Keep your invoice: it is your warranty card.</p>
<h2>Warranty periods</h2>
<p>Kitchen appliances (mixer grinders, induction cooktops, kettles): 24 months.</p>
<p>Phones and tablets: 12 months. Laptops: 12 months.</p>
<p>Televisions: 12 months on the product and 36 months on the display panel.</p>
<p>Earbuds and headphones: 6 months.</p>
<h2>Nimbu Care+ extended warranty</h2>
<p>Laptops can be covered for 24 extra months with Nimbu Care+ for Rs. 2,499.
Nimbu Care+ must be bought within 30 days of delivery.</p>
<h2>How to claim</h2>
<p>Raise a claim in the app under Orders, Warranty claim. A technician visit is booked
within 3 working days. Physical damage and liquid damage are not covered.</p>
"""


def _render_page_as_scan(html: str, dpi: int = 150) -> bytes:
    """Make a 'scanned' page image: render HTML to an image, then make it look scanned."""
    tmp = pymupdf.open()
    _page_with_html(tmp, html, header=None, footer=None)
    pix = tmp[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    img = img.rotate(0.6, expand=False, fillcolor=255)  # a slightly crooked scan
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))  # a slightly soft scan
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)  # scanners usually produce JPEG; also keeps the file small
    return buf.getvalue()


def make_scanned_pdf() -> Path:
    png = _render_page_as_scan(WARRANTY_HTML)
    doc = pymupdf.open()
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_image(page.rect, stream=png)  # image only, NO text layer
    out = RAW / "warranty_terms_scanned.pdf"
    _save(doc, out)
    return out


# ---------------------------------------------------------------- 3. mixed PDF
VENDOR_P1 = """
<h1>Vendor Onboarding Guide</h1>
<p>This guide is for the Procurement team and for sellers who want to list products on
Nimbu Mart. Onboarding is completed within 7 business days after all documents are
received.</p>
<h2>Payment terms</h2>
<p>Vendors are paid on Net-45 terms: 45 days after the invoice is accepted. Payments are made
by NEFT every Friday.</p>
"""
VENDOR_P2_TEXT = """
<h2>Document checklist</h2>
<p>The checklist below is the same one printed on the paper form. It is included as an image
because many vendors still send scanned forms.</p>
"""
VENDOR_CHECKLIST_IMG_HTML = """
<h2>Mandatory documents</h2>
<p>1. GST registration certificate (GSTIN)</p>
<p>2. PAN card of the business</p>
<p>3. Cancelled cheque for bank verification</p>
<p>4. Signed vendor agreement</p>
<p>5. FSSAI licence (only for food sellers)</p>
"""


def make_mixed_pdf() -> Path:
    doc = pymupdf.open()
    _page_with_html(doc, VENDOR_P1, header="Nimbu Mart | Procurement", footer="Page 1 of 2")
    p2 = _page_with_html(doc, VENDOR_P2_TEXT, header="Nimbu Mart | Procurement", footer="Page 2 of 2")

    # Render the checklist as a picture and paste it into page 2 (below the digital text).
    tmp = pymupdf.open()
    t = tmp.new_page(width=420, height=230)
    t.insert_htmlbox(pymupdf.Rect(10, 10, 410, 220), VENDOR_CHECKLIST_IMG_HTML, css=CSS)
    pix = t.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    img_rect = pymupdf.Rect(MARGIN, 220, MARGIN + 420, 220 + 230)
    p2.insert_image(img_rect, stream=pix.tobytes("jpg", jpg_quality=75))
    out = RAW / "vendor_onboarding_mixed.pdf"
    _save(doc, out)
    return out


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for fn in (make_travel_pdf, make_scanned_pdf, make_mixed_pdf):
        print("created", fn().relative_to(RAW.parents[1]))
