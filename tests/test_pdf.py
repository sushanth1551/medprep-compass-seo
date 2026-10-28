"""PDF export tests. Local sample text only; no Groq request, no network."""

import io
import json

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
from reportlab.platypus import ListFlowable, Paragraph

import app.web as web
from app.pdf_export import build_story, clean_text, inline_markup, markdown_to_pdf, pdf_filename
from app.providers.groq_provider import GroqProvider

SAMPLE = """# Sample Study Guide

> REVIEWED DRAFT - needs human review.

**Meta description:** A short guide with *emphasis* and `code`.

## Introduction

First paragraph with a line
that continues here. See https://www.usmle.org/step-exams/step-1 for details.

## Steps

- First bullet
- Second bullet with **bold** text

1. First numbered
2. Second numbered

## Frequently Asked Questions

### How long should I study?

It depends on you ‑ non‑breaking hyphen and an em dash — should print.

## References

1. USMLE. Step 1. https://www.usmle.org/step-exams/step-1
"""


def pdf_text(data):
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages)


# Conversion ------------------------------------------------------------------------------------
def test_pdf_is_a_real_pdf_with_content():
    data = markdown_to_pdf(SAMPLE)
    assert data.startswith(b"%PDF-")
    reader = PdfReader(io.BytesIO(data))
    assert len(reader.pages) >= 1
    assert reader.metadata.title == "Sample Study Guide"


def test_pdf_text_has_headings_lists_faq_and_no_raw_markdown():
    text = pdf_text(markdown_to_pdf(SAMPLE))
    for needle in ("Sample Study Guide", "Introduction", "Steps", "First bullet", "First numbered",
                   "Frequently Asked Questions", "How long should I study?", "References",
                   "emphasis", "bold", "Page 1"):
        assert needle in text
    for raw in ("##", "**", "`", "](http"):
        assert raw not in text
    assert "em dash" in text and "non-breaking hyphen" in text  # special characters survive


def test_story_structure_uses_heading_paragraph_and_list_flowables():
    story = build_story(SAMPLE)
    styles = [f.style.name for f in story if isinstance(f, Paragraph)]
    assert styles[0] == "DocTitle" and "H2" in styles and "H3" in styles and "Quote" in styles
    lists = [f for f in story if isinstance(f, ListFlowable)]
    assert len(lists) == 3  # bullets, numbered list, references


def test_urls_become_clickable_links():
    assert 'href="https://www.usmle.org/step-exams/step-1"' in inline_markup(
        "see https://www.usmle.org/step-exams/step-1.")
    assert 'href="https://a.example/x"' in inline_markup("[label](https://a.example/x)")
    reader = PdfReader(io.BytesIO(markdown_to_pdf(SAMPLE)))
    links = [a.get_object().get("/A", {}).get("/URI") for p in reader.pages
             for a in (p.get("/Annots") or [])]
    assert "https://www.usmle.org/step-exams/step-1" in links


def test_text_is_escaped_so_markup_in_articles_cannot_break_the_pdf():
    assert "&lt;b&gt;" in inline_markup("5 < 6 and <b>raw</b> & more")
    markdown_to_pdf("# T\n\nText with <unclosed tag & ampersand")  # must not raise


def test_long_article_breaks_across_pages():
    body = "\n\n".join(f"## Section {i}\n\n" + ("Sentence about studying. " * 60) for i in range(30))
    reader = PdfReader(io.BytesIO(markdown_to_pdf("# Long\n\n" + body)))
    assert len(reader.pages) > 3
    assert "Page 2" in reader.pages[1].extract_text()


def test_empty_article_is_rejected():
    with pytest.raises(ValueError):
        markdown_to_pdf("   \n")


def test_unsupported_characters_are_replaced_not_crashing():
    assert clean_text("a‑b c") == "a-b c"
    markdown_to_pdf("# T\n\nChinese 中文 and emoji \U0001f600 text")


@pytest.mark.parametrize("title, expected", [
    ("How to Prepare for USMLE Step 1: A Guide!", "how-to-prepare-for-usmle-step-1-a-guide.pdf"),
    ('../../evil"name\\x', "evil-name-x.pdf"),
    ("", "article.pdf"),
])
def test_filenames_are_safe(title, expected):
    assert pdf_filename(title) == expected


# Web route ------------------------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(web, "METRICS_DIR", tmp_path / "runs")
    monkeypatch.setenv("AI_PROVIDER", "mock")
    return TestClient(web.app, follow_redirects=False)


def make_article(client):
    response = client.post("/generate", data={"topic": "Pdf test topic", "keyword": "", "length": "short"})
    return response.headers["location"].rsplit("/", 1)[1]


def test_pdf_download_route_returns_real_pdf_of_the_saved_article(client):
    slug = make_article(client)
    md_before = (web.OUTPUT_DIR / f"{slug}.md").read_text(encoding="utf-8")
    page = client.get(f"/articles/{slug}")
    assert "Download PDF" in page.text and "Download Markdown" in page.text
    response = client.get(f"/articles/{slug}/download.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == f'attachment; filename="{slug}.pdf"'
    assert response.content.startswith(b"%PDF-")
    text = pdf_text(response.content)
    assert "Pdf test topic" in text and "Frequently Asked Questions" in text
    assert "DEMO CONTENT" in text
    assert (web.OUTPUT_DIR / f"{slug}.md").read_text(encoding="utf-8") == md_before  # untouched
    assert len(list(web.OUTPUT_DIR.glob("*.md"))) == 1  # the PDF did not create another article


def test_markdown_download_still_works(client):
    slug = make_article(client)
    response = client.get(f"/articles/{slug}/download")
    assert response.status_code == 200 and response.text.startswith("# ")
    assert "markdown" in response.headers["content-type"]


@pytest.mark.parametrize("path", ["/articles/missing/download.pdf", "/articles/BAD_NAME/download.pdf"])
def test_pdf_for_unknown_article_is_404(client, path):
    assert client.get(path).status_code == 404


def test_pdf_never_contains_secrets_or_run_settings(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-secret-should-never-appear")
    slug = make_article(client)
    text = pdf_text(client.get(f"/articles/{slug}/download.pdf").content)
    assert "gsk-secret" not in text and "GROQ_API_KEY" not in text


def test_truncated_generation_gives_no_article_and_no_pdf(client, monkeypatch):
    from types import SimpleNamespace

    choice = SimpleNamespace(message=SimpleNamespace(content='{"title": "cut'), finish_reason="length")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=99, prompt_tokens_details=None,
                            completion_tokens_details=None)
    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[choice], usage=usage))))
    monkeypatch.setattr(web, "build_provider", lambda **kw: GroqProvider("k", "m", client=stub, **kw))
    response = client.post("/generate", data={"topic": "Cut off", "keyword": "", "length": "medium"})
    assert response.status_code == 502 and "cut off" in response.text
    assert not list(web.OUTPUT_DIR.glob("*")) if web.OUTPUT_DIR.exists() else True
    assert client.get("/articles/cut-off/download.pdf").status_code == 404
    failed = json.loads(next(web.METRICS_DIR.iterdir()).read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
