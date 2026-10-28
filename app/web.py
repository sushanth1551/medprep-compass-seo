"""Simple local web interface. Start with: python -m app.web

It calls the same workflow, providers, metrics and SEO checks as the command line.
API keys stay on the server: no template or response ever contains them.
Articles are saved as files in output/ (never overwritten); nothing is published.
"""

import re
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from app import store
from app.config import (
    build_provider,
    get_max_output_tokens,
    get_provider_name,
    load_local_env,
    output_limit_warning,
    regenerate_enabled,
)
from app.metrics import DEFAULT_RUNS_DIR
from app.models import ConfigError, ProviderError, ValidationError
from app.pdf_export import markdown_to_pdf, pdf_filename
from app.providers.prompts import ARTICLE_LENGTHS
from app.quality import build_quality_report, can_approve_export
from app.revision import revise_section
from app.seo_checks import run_seo_checks
from app.sources import check_sources
from app.workflow import DEFAULT_OUTPUT_DIR, run_workflow

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
METRICS_DIR = DEFAULT_RUNS_DIR
LENGTHS = {name: words for name, (words, _label) in ARTICLE_LENGTHS.items()}
SLUG_PATTERN = store.SLUG_PATTERN

app = FastAPI(title="MedPrep Compass - SEO Draft Generator")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
# Only one paid request at a time, so a double click cannot start two.
_generation_lock = threading.Lock()


# Display helpers ------------------------------------------------------------------------
_LINK = re.compile(r"(https?://[^\s<>&]+[^\s<>&.,;:!?)])")


def _inline(text):
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return _LINK.sub(r'<a href="\1" rel="noopener noreferrer">\1</a>', text)


def render_body(text):
    """Show a section body as HTML (paragraphs, ### sub-headings, lists). Text is escaped first."""
    out, para, items, tag = [], [], [], None

    def flush():
        nonlocal items, tag
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()
        if items:
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + f"</{tag}>")
            items, tag = [], None

    for line in (text or "").splitlines():
        stripped = line.strip()
        bullet = re.match(r"^[-*+]\s+(.*)", stripped)
        number = re.match(r"^\d+[.)]\s+(.*)", stripped)
        if not stripped:
            flush()
        elif stripped.startswith("### "):
            flush()
            out.append(f"<h4>{_inline(stripped[4:])}</h4>")
        elif bullet or number:
            want = "ul" if bullet else "ol"
            if tag != want:
                flush()
                tag = want
            items.append((bullet or number).group(1))
        else:
            if items:
                flush()
            para.append(stripped)
    flush()
    return Markup("".join(out))


templates.env.filters["body_html"] = render_body


def _provider_summary():
    """Provider/model for display. Never includes the key."""
    try:
        provider = build_provider()
    except ConfigError as exc:
        return None, str(exc)
    return provider, None


def _recent_articles(limit=40):
    if not Path(OUTPUT_DIR).exists():
        return []
    items = []
    for md in sorted(Path(OUTPUT_DIR).glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        title = md.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip() or md.stem
        record = store.load_record(OUTPUT_DIR, md.stem)
        items.append({"slug": md.stem, "title": title, "full": record is not None,
                      "revision": record["revision"] if record else None})
    return items[:limit]


def _lengths_for_form():
    return {name: label for name, (_, label) in ARTICLE_LENGTHS.items()}


def _render_index(request, status_code=200, error=None, form=None):
    provider, config_error = _provider_summary()
    form = form or {"topic": "", "keyword": "", "length": "medium"}
    limit_warning = None
    if provider and form.get("length") in LENGTHS:
        limit_warning = output_limit_warning(get_max_output_tokens(), LENGTHS[form["length"]])
    context = {
        "error": error,
        "config_error": config_error,
        "provider_name": provider.name if provider else None,
        "provider_id": get_provider_name() if provider else None,
        "articles": _recent_articles(),
        "lengths": _lengths_for_form(),
        "form": form,
        "limit_warning": limit_warning,
    }
    return templates.TemplateResponse(request, "index.html", context, status_code=status_code)


def _find(slug):
    if not SLUG_PATTERN.match(slug):
        return None
    md = store.md_path(OUTPUT_DIR, slug)
    return md if md.is_file() else None


def _article_page(request, slug, error=None, notice=None, status_code=200):
    md = _find(slug)
    if md is None:
        return _render_index(request, 404, "Article not found.")
    record = store.load_record(OUTPUT_DIR, slug)
    context = {"slug": slug, "data": record, "error": error, "notice": notice,
               "raw_markdown": None if record else md.read_text(encoding="utf-8")}
    if record:
        problems = store.consistency_problems(OUTPUT_DIR, record)
        stale = store.is_stale(problems)
        context.update(
            problems=problems,
            stale=stale,
            quality=build_quality_report(record, stale=stale),
            revisions=store.revisions_of(OUTPUT_DIR, record["base_slug"]),
            can_approve=can_approve_export(record, stale)[0] and not problems,
        )
    return templates.TemplateResponse(request, "article.html", context, status_code=status_code)


# Routes -----------------------------------------------------------------------------------
@app.get("/health")
def health():
    """For deployment checks. Reveals nothing about configuration."""
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
def index(request: Request, topic: str = "", keyword: str = "", length: str = "medium"):
    """Optional query values pre-fill the form (used by "start a new draft with the same topic")."""
    form = {"topic": topic[:120], "keyword": keyword[:120], "length": length if length in LENGTHS else "medium"}
    return _render_index(request, form=form)


@app.post("/generate")
def generate(request: Request, topic: str = Form(""), keyword: str = Form(""),
             length: str = Form("medium")):
    form = {"topic": topic, "keyword": keyword, "length": length}
    if length not in LENGTHS:
        return _render_index(request, 400, "Choose a valid article length.", form)
    if not _generation_lock.acquire(blocking=False):
        return _render_index(request, 429, "A generation is already running. Wait for it to finish.", form)
    try:
        provider = build_provider(target_words=LENGTHS[length])
        result = run_workflow(
            topic, keyword, provider=provider, output_dir=OUTPUT_DIR, metrics_dir=METRICS_DIR,
            regenerate_failed=regenerate_enabled(), diagnostics_dir=Path(METRICS_DIR).parent / "diagnostics",
        )
        slug = Path(result.output_path).stem
        markdown = Path(result.output_path).read_text(encoding="utf-8")
        record = store.new_record(slug, result.article, result.check_results, result.seo_report,
                                  result.metrics, length, markdown)
        store.save_record(OUTPUT_DIR, record)
    except ValidationError as exc:
        return _render_index(request, 400, f"Input problem: {exc}", form)
    except ConfigError as exc:
        return _render_index(request, 500, f"Configuration problem: {exc}", form)
    except ProviderError as exc:
        # No fallback to mock output and no automatic retry.
        return _render_index(request, 502, f"The AI provider failed: {exc}", form)
    except OSError as exc:
        return _render_index(request, 500, f"File problem: {exc}", form)
    finally:
        _generation_lock.release()
    return RedirectResponse(f"/articles/{slug}", status_code=303)  # refresh won't resubmit


@app.get("/articles/{slug}", response_class=HTMLResponse)
def view_article(request: Request, slug: str):
    return _article_page(request, slug)


def _export_blocked(slug):
    """Return an error response if this revision must not be exported as a finished article."""
    record = store.load_record(OUTPUT_DIR, slug)
    if record and not record["metrics"].get("appears_complete"):
        reasons = " ".join(record["metrics"].get("incomplete_reasons", []))
        return HTMLResponse(
            f"This article looks incomplete, so it cannot be exported as a finished article. {escape(reasons)}",
            status_code=409,
        )
    return None


def _export_headers(slug):
    record = store.load_record(OUTPUT_DIR, slug)
    return {"X-Article-Run-Id": record["run_id"], "X-Article-Revision": str(record["revision"])} if record else {}


@app.get("/articles/{slug}/download")
def download_article(slug: str):
    md = _find(slug)
    if md is None:
        return HTMLResponse("Article not found.", status_code=404)
    blocked = _export_blocked(slug)
    if blocked:
        return blocked
    return FileResponse(md, media_type="text/markdown", filename=f"{slug}.md", headers=_export_headers(slug))


@app.get("/articles/{slug}/download.pdf")
def download_pdf(slug: str):
    """PDF of the saved article. Built from the saved file; never triggers a new generation."""
    md = _find(slug)
    if md is None:
        return HTMLResponse("Article not found.", status_code=404)
    blocked = _export_blocked(slug)
    if blocked:
        return blocked
    try:
        pdf = markdown_to_pdf(md.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return HTMLResponse(f"Could not create the PDF: {exc}", status_code=500)
    headers = {"Content-Disposition": f'attachment; filename="{pdf_filename(slug)}"', **_export_headers(slug)}
    return Response(pdf, media_type="application/pdf", headers=headers)


@app.post("/articles/{slug}/check-sources")
def check_article_sources(request: Request, slug: str):
    """Fetch the cited pages (free, no AI). Records what was reachable; verifies nothing."""
    record = store.load_record(OUTPUT_DIR, slug) if _find(slug) else None
    if record is None:
        return _render_index(request, 404, "Article not found.")
    record["sources"] = check_sources(store.article_from_record(record))
    store.save_record(OUTPUT_DIR, record)
    return RedirectResponse(f"/articles/{slug}", status_code=303)


@app.post("/articles/{slug}/recheck")
def recheck_article(request: Request, slug: str):
    """Re-run the free automatic checks on the SAVED text (no AI). The article file is untouched."""
    record = store.load_record(OUTPUT_DIR, slug) if _find(slug) else None
    if record is None:
        return _render_index(request, 404, "Article not found.")
    article = store.article_from_record(record)
    target = LENGTHS.get(record["length"])
    seo = run_seo_checks(article, target_words=target if record["metrics"]["provider"] != "mock" else None)
    record["seo"] = {**asdict(seo), "score_text": seo.score_text}
    record["checks_rerun_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.save_record(OUTPUT_DIR, record)
    return RedirectResponse(f"/articles/{slug}", status_code=303)


@app.post("/articles/{slug}/revise")
def revise_article(request: Request, slug: str, section: int = Form(-1)):
    """Rewrite ONE section as a new revision. One paid model call, only on this explicit click."""
    record = store.load_record(OUTPUT_DIR, slug) if _find(slug) else None
    if record is None:
        return _render_index(request, 404, "Article not found.")
    if not _generation_lock.acquire(blocking=False):
        return _article_page(request, slug, error="A generation is already running.", status_code=429)
    try:
        provider = build_provider(target_words=LENGTHS.get(record["length"]))
        new = revise_section(record, section, provider, OUTPUT_DIR, METRICS_DIR)
        store.save_record(OUTPUT_DIR, new)
    except (ValidationError, ConfigError) as exc:
        return _article_page(request, slug, error=str(exc), status_code=400)
    except ProviderError as exc:
        return _article_page(request, slug, error=f"The AI provider failed: {exc}", status_code=502)
    except OSError as exc:
        return _article_page(request, slug, error=f"File problem: {exc}", status_code=500)
    finally:
        _generation_lock.release()
    return RedirectResponse(f"/articles/{new['slug']}", status_code=303)


@app.post("/articles/{slug}/approve")
def approve_article(request: Request, slug: str, level: str = Form(""), confirm: str = Form("")):
    """Record a human decision for this exact revision. Nothing is published by the app."""
    record = store.load_record(OUTPUT_DIR, slug) if _find(slug) else None
    if record is None:
        return _render_index(request, 404, "Article not found.")
    problems = store.consistency_problems(OUTPUT_DIR, record)
    allowed, reason = can_approve_export(record, store.is_stale(problems))
    if problems or not allowed:
        return _article_page(request, slug, error="Cannot approve: " + (reason or " ".join(problems)),
                             status_code=409)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if level == "export":
        record["approvals"]["export"] = {"at": now}
    elif level == "publication":
        if not record["approvals"].get("export"):
            return _article_page(request, slug, error="Approve for export first.", status_code=409)
        if confirm != "yes":
            return _article_page(request, slug, error="Tick the box to confirm you checked the facts and sources.",
                                 status_code=400)
        record["approvals"]["publication"] = {"at": now, "confirmed_facts_and_sources_checked": True}
    else:
        return _article_page(request, slug, error="Unknown approval type.", status_code=400)
    store.save_record(OUTPUT_DIR, record)
    return RedirectResponse(f"/articles/{slug}", status_code=303)


if __name__ == "__main__":
    import uvicorn

    load_local_env()
    uvicorn.run(app, host="127.0.0.1", port=8000)
