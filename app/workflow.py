"""The six workflow steps, each in its own small function."""

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from app.metrics import build_run_metrics, check_completeness, save_run_metrics
from app.models import (
    DEMO_NOTICE,
    Article,
    OutlineSection,
    ProviderError,
    ValidationError,
    WorkflowResult,
)
from app.pricing import compute_cost
from app.providers.mock_provider import MockProvider
from app.seo_checks import run_seo_checks
from app.token_tracker import TokenTracker

MAX_TOPIC_LENGTH = 120
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


# Step 1 ---------------------------------------------------------------
def validate_inputs(topic, keyword=None):
    """Return a cleaned (topic, keyword) pair or raise ValidationError."""
    # Remove invisible BOM characters (PowerShell can add one to piped input).
    topic = (topic or "").replace("﻿", "").strip()
    keyword = (keyword or "").replace("﻿", "").strip()
    if not topic:
        raise ValidationError("Topic cannot be empty.")
    if len(topic) > MAX_TOPIC_LENGTH:
        raise ValidationError(f"Topic is too long (max {MAX_TOPIC_LENGTH} characters).")
    if not re.search(r"[A-Za-z0-9]", topic):
        raise ValidationError("Topic must contain at least one letter or number.")
    if len(keyword) > MAX_TOPIC_LENGTH:
        raise ValidationError(f"Keyword is too long (max {MAX_TOPIC_LENGTH} characters).")
    return topic, keyword or topic


# Step 2 ---------------------------------------------------------------
def create_outline(topic, keyword):
    return [
        OutlineSection("Introduction", f"introduce {keyword} and who this guide is for"),
        OutlineSection("Key Concepts to Know", f"explain the main ideas a learner needs for {topic}"),
        OutlineSection("A Practical Study Plan", "suggest a step-by-step way to study"),
        OutlineSection("Common Mistakes to Avoid", "list typical study pitfalls"),
        OutlineSection("Conclusion", "summarise and point to next steps"),
    ]


# Steps 3 and 4 --------------------------------------------------------
def build_article(topic, keyword, outline, provider, run_id=''):
    sections = provider.generate_sections(topic, keyword, outline)
    faq = provider.generate_faq(topic, keyword)
    if hasattr(provider, "generate_metadata"):
        title, meta = provider.generate_metadata(topic, keyword, outline)
    else:
        title = f"{topic}: A Study Guide for Medical Exam Candidates"
        meta = f"A demo outline-style guide about {keyword} for medical exam preparation."
    return Article(
        title=title,
        meta_description=meta[:160],
        keyword=keyword,
        sections=sections,
        faq=faq,
        provider_name=provider.name,
        notice=getattr(provider, "notice", DEMO_NOTICE),
        run_id=run_id,
        **_extras(provider),
    )


def completeness_target(provider, target):
    """The mock writes fixed placeholder text, so only real providers are held to the length target."""
    return target if hasattr(provider, "call_log") else None


def _extras(provider):
    """Optional structured fields (search intent, audience, references, review flags)."""
    data = provider.extras() if hasattr(provider, "extras") else {}
    return {k: v for k, v in data.items() if v}


# Step 5 ---------------------------------------------------------------
def run_checks(article):
    """Basic structure checks only. These do not judge quality or accuracy."""
    return {
        "has_title": bool(article.title.strip()),
        "meta_description_present": bool(article.meta_description.strip()),
        "meta_description_max_160_chars": len(article.meta_description) <= 160,
        "has_at_least_3_headings": len(article.sections) >= 3,
        "has_faq": len(article.faq) >= 1,
        "has_review_notice": article.notice in article.to_markdown(),
    }


# Step 6 ---------------------------------------------------------------
def safe_filename(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60].strip("-")
    return (slug or "article") + ".md"


def unique_path(output_dir, filename):
    """Never overwrite an existing draft: add -2, -3, ... if the name is taken."""
    path = Path(output_dir) / filename
    stem, n = path.stem, 2
    while path.exists():
        path = path.with_name(f"{stem}-{n}{path.suffix}")
        n += 1
    return path


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_draft(article, output_dir, report=None, filename=None):
    """Save the Markdown draft; return (draft_path, report_path or None).

    The report file records the run id, revision and a hash of the saved Markdown so a stale
    or mismatched report can be detected later.
    """
    output_dir = Path(output_dir)
    report_path = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = unique_path(output_dir, filename or safe_filename(article.title))
        markdown = article.to_markdown()
        path.write_text(markdown, encoding="utf-8")
        if report is not None:
            report_path = path.with_suffix(".seo-report.json")
            record = {**asdict(report), "run_id": article.run_id, "revision": article.revision,
                      "markdown_file": path.name, "markdown_sha256": text_hash(markdown)}
            report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not save the draft in '{output_dir}': {exc}") from exc
    return path, report_path


def regenerate_empty_sections(article, topic, keyword, provider, limit):
    """Targeted fix: rewrite only sections whose body is empty, at most `limit` of them.

    Never rewrites the whole article. Each rewrite is one counted model call.
    """
    done = []
    if not hasattr(provider, "regenerate_section"):
        return done
    for i, (heading, body) in enumerate(article.sections):
        if len(done) >= limit:
            break
        if not body.strip():
            article.sections[i] = (heading, provider.regenerate_section(topic, keyword, heading))
            done.append(heading)
    return done


def _usage_for(provider, tracker, topic, keyword, outline, article):
    """Fill the tracker: actual usage for real providers, an estimate for the mock."""
    if hasattr(provider, "call_log"):
        totals = provider.usage_totals()
        if totals:
            tracker.record_actual(
                totals["input_tokens"], totals["output_tokens"], totals["cached_input_tokens"]
            )
        else:
            tracker.mark_unavailable()  # real provider, no usage numbers: do not invent any
        return
    if hasattr(provider, "build_prompt_text"):
        prompt_text = provider.build_prompt_text(topic, keyword, outline)
    else:
        lines = [f"{o.heading}: {o.purpose}" for o in outline]
        prompt_text = "\n".join([topic, keyword] + lines)
    tracker.record_estimate(prompt_text, article.to_markdown())


def save_diagnostics(provider, run_id, diagnostics_dir):
    """Keep the facts about an unusual model reply (finish reason, raw reply text, token counts).

    Contains only the model's article text and counters - never prompts, keys or settings.
    """
    events = getattr(provider, "diagnostics", None)
    if not events or diagnostics_dir is None:
        return ""
    folder = Path(diagnostics_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"diag-{run_id}.json"
    with open(path, "x", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "events": events}, handle, indent=2)
    return str(path)


def call_facts(provider, diagnostics_path):
    """Per-run facts for the metrics file: why each reply ended and how many follow-ups were made."""
    return {
        "finish_reasons": [c.get("finish_reason") for c in getattr(provider, "call_log", [])],
        "continuation_calls": getattr(provider, "continuations_used", 0),
        "diagnostics_file": diagnostics_path,
    }


def _failed_usage(provider):
    """Whatever usage the API reported before the failure (None if it reported none)."""
    log = [c for c in getattr(provider, "call_log", []) if c["input_tokens"] is not None]
    if not log:
        return None
    tracker = TokenTracker()
    tracker.record_actual(
        sum(c["input_tokens"] for c in log), sum(c["output_tokens"] or 0 for c in log)
    )
    return tracker.summary()


def run_workflow(
    topic,
    keyword=None,
    provider=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    tracker=None,
    metrics_dir=None,
    regenerate_failed=False,
    max_regenerations=1,
    pricing=None,
    diagnostics_dir=None,
):
    """Run the six steps. If `metrics_dir` is given, a unique run-metrics file is saved."""
    provider = provider or MockProvider()
    topic, keyword = validate_inputs(topic, keyword)
    outline = create_outline(topic, keyword)
    run_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    regenerated = []
    try:
        article = build_article(topic, keyword, outline, provider, run_id)
        if regenerate_failed:
            regenerated = regenerate_empty_sections(
                article, topic, keyword, provider, max_regenerations
            )
    except ProviderError as exc:
        if metrics_dir is not None:  # a failed paid attempt is still worth recording
            failed = build_run_metrics(
                topic=topic, keyword=keyword, provider=provider, usage=_failed_usage(provider),
                call_count=getattr(provider, "call_count", 0),
                elapsed_seconds=time.perf_counter() - started, seo_report=None,
                complete=False, incomplete_reasons=["Generation failed."],
                cost=compute_cost(provider.provider_id, provider.model, "unavailable", 0, 0),
                status="failed", error=str(exc), run_id=run_id,
                extra=call_facts(provider, save_diagnostics(provider, run_id, diagnostics_dir)),
            )
            save_run_metrics(failed, metrics_dir)
        raise
    elapsed = time.perf_counter() - started

    checks = run_checks(article)
    target = getattr(provider, "target_words", None)
    seo_report = run_seo_checks(article, target_words=target)  # deterministic: no extra model call
    complete, reasons = check_completeness(article, outline, completeness_target(provider, target))
    path, report_path = save_draft(article, output_dir, seo_report)

    tracker = tracker or TokenTracker()
    _usage_for(provider, tracker, topic, keyword, outline, article)
    usage = tracker.summary()
    if usage.usage_source == "actual":
        cost = compute_cost(
            provider.provider_id, provider.model, "actual", usage.actual_input_tokens,
            usage.actual_output_tokens, usage.cached_input_tokens, pricing,
        )
    else:
        cost = compute_cost(
            getattr(provider, "provider_id", "unknown"), getattr(provider, "model", None),
            usage.usage_source, 0, 0, None, pricing,
        )
    metrics = build_run_metrics(
        topic=topic, keyword=keyword, provider=provider, usage=usage,
        call_count=getattr(provider, "call_count", 0), elapsed_seconds=elapsed,
        seo_report=seo_report, complete=complete, incomplete_reasons=reasons,
        cost=cost, regenerated_sections=regenerated, run_id=run_id,
        extra=call_facts(provider, save_diagnostics(provider, run_id, diagnostics_dir)),
        artifacts={
            "article_slug": path.stem,
            "revision": article.revision,
            "draft_path": str(path),
            "seo_report_path": str(report_path or ""),
            "markdown_sha256": text_hash(path.read_text(encoding="utf-8")),
        },
    )
    metrics_path = save_run_metrics(metrics, metrics_dir) if metrics_dir is not None else ""
    return WorkflowResult(
        article, str(path), usage, checks, seo_report, str(report_path or ""),
        metrics, str(metrics_path),
    )
