"""Turn a saved Markdown draft into a readable PDF (ReportLab, no network, no AI call).

Handles the Markdown the app produces: headings, paragraphs, bullet and numbered lists,
block quotes, **bold**, *italic*, `code`, links and bare URLs (made clickable).
The PDF is built from the saved article text only; nothing else (no settings, no keys).
"""

import io
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

# Built-in PDF fonts only cover Windows-1252; swap look-alike characters, drop the rest.
_CHAR_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "→": "->", "≤": "<=", "≥": ">=", "✓": "v",
}
_INLINE = re.compile(
    r"\[(?P<ltext>[^\]]+)\]\((?P<lurl>https?://[^)\s]+)\)"
    r"|(?P<url>https?://[^\s<>)\]]+[^\s<>)\].,;:!?])"
    r"|\*\*(?P<bold>.+?)\*\*"
    r"|\*(?P<italic>[^*\s][^*]*?)\*"
    r"|`(?P<code>[^`]+)`"
)
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)")
_NUMBER = re.compile(r"^\s*(\d+)[.)]\s+(.*)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)")


def clean_text(text):
    """Make text safe for the built-in PDF fonts."""
    for old, new in _CHAR_MAP.items():
        text = text.replace(old, new)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def inline_markup(text):
    """Escape text and convert inline Markdown to ReportLab's mini-HTML."""
    text = clean_text(text)
    out, pos = [], 0
    for m in _INLINE.finditer(text):
        out.append(escape(text[pos:m.start()]))
        if m.group("lurl"):
            out.append(_link(m.group("lurl"), m.group("ltext")))
        elif m.group("url"):
            out.append(_link(m.group("url"), m.group("url")))
        elif m.group("bold"):
            out.append(f"<b>{escape(m.group('bold'))}</b>")
        elif m.group("italic"):
            out.append(f"<i>{escape(m.group('italic'))}</i>")
        else:
            out.append(f'<font face="Courier">{escape(m.group("code"))}</font>')
        pos = m.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def _link(url, label):
    return f'<a href="{escape(url, {chr(34): "&quot;"})}" color="#1a4f9c">{escape(label)}</a>'


def _styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=10.5,
                          leading=15, spaceAfter=6, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("DocTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=21, leading=26, spaceAfter=12, alignment=TA_LEFT),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=14.5,
                             leading=18, spaceBefore=14, spaceAfter=6, keepWithNext=1),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=12,
                             leading=15, spaceBefore=10, spaceAfter=4, keepWithNext=1),
        "body": body,
        "quote": ParagraphStyle("Quote", parent=body, leftIndent=10, backColor=colors.HexColor("#fff6dd"),
                                borderPadding=6, spaceBefore=4, spaceAfter=10, fontSize=9.5, leading=13),
        "item": body,
    }


def build_story(markdown_text):
    """Convert Markdown text to a list of ReportLab flowables."""
    st = _styles()
    story, para, items, kind, start = [], [], [], None, 1

    def flush_para():
        if para:
            story.append(Paragraph(inline_markup(" ".join(para)), st["body"]))
            para.clear()

    def flush_list():
        nonlocal items, kind
        if items:
            flow = [ListItem(Paragraph(inline_markup(t), st["item"]), leftIndent=18) for t in items]
            if kind == "number":
                story.append(ListFlowable(flow, bulletType="1", start=start, bulletFontName="Helvetica",
                                          bulletFontSize=10.5, leftIndent=22))
            else:
                story.append(ListFlowable(flow, bulletType="bullet", start="•", bulletFontSize=10.5,
                                          leftIndent=22))
            story.append(Spacer(1, 4))
        items, kind = [], None

    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        heading, bullet, number = _HEADING.match(line), _BULLET.match(line), _NUMBER.match(line)
        if not line.strip():
            flush_para()
            flush_list()
        elif heading:
            flush_para()
            flush_list()
            level = len(heading.group(1))
            style = "title" if level == 1 else "h2" if level == 2 else "h3"
            story.append(Paragraph(inline_markup(heading.group(2)), st[style]))
        elif line.lstrip().startswith(">"):
            flush_para()
            flush_list()
            story.append(Paragraph(inline_markup(line.lstrip()[1:].strip()), st["quote"]))
        elif bullet or number:
            flush_para()
            new_kind = "bullet" if bullet else "number"
            if kind != new_kind:
                flush_list()
                kind = new_kind
                start = int(number.group(1)) if number else 1
            items.append(bullet.group(1) if bullet else number.group(2))
        elif raw.startswith("  ") and items:
            items[-1] += " " + line.strip()  # continuation of a list item
        else:
            flush_list()
            para.append(line.strip())
    flush_para()
    flush_list()
    return story


def _page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def markdown_to_pdf(markdown_text, title=None):
    """Return the PDF file as bytes."""
    if not markdown_text or not markdown_text.strip():
        raise ValueError("There is no article text to put in a PDF.")
    if title is None:
        match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
        title = match.group(1).strip() if match else "Article"
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm,
        bottomMargin=22 * mm, title=clean_text(title), author="MedPrep Compass",
        subject="Draft article - requires human review",
    )
    doc.build(build_story(markdown_text), onFirstPage=_page_number, onLaterPages=_page_number)
    return buffer.getvalue()


def pdf_filename(title_or_slug):
    """A safe ASCII file name (letters, digits, hyphens) ending in .pdf."""
    slug = re.sub(r"[^a-z0-9]+", "-", clean_text(title_or_slug or "").lower()).strip("-")[:80].strip("-")
    return (slug or "article") + ".pdf"
