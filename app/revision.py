"""Controlled revision: rewrite ONE section as a NEW revision. One visible, paid model call.

The original revision is never touched. Checks, completeness and metrics are recalculated for
the new file, and sources/approvals start empty again because they belonged to the old text.
"""

import time
import uuid

from app.metrics import build_run_metrics, check_completeness, save_run_metrics
from app.models import ProviderError, ValidationError
from app.pricing import compute_cost
from app.seo_checks import run_seo_checks
from app.store import article_from_record, new_record, next_revision
from app.token_tracker import TokenTracker
from app.workflow import (
    _failed_usage,
    _usage_for,
    completeness_target,
    create_outline,
    run_checks,
    save_draft,
    text_hash,
)


def revise_section(record, section_index, provider, output_dir, metrics_dir, pricing=None):
    """Return the new revision's record (already saved to disk, with its own metrics file)."""
    if not hasattr(provider, "regenerate_section"):
        raise ValidationError("This provider cannot rewrite a single section.")
    article = article_from_record(record)
    if not 0 <= section_index < len(article.sections):
        raise ValidationError("That section does not exist.")
    topic, keyword = record["metrics"]["topic"], record["metrics"]["keyword"]
    heading = article.sections[section_index][0]
    base_slug = record["base_slug"]
    run_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        body = provider.regenerate_section(topic, keyword, heading)
    except ProviderError as exc:
        failed = build_run_metrics(
            topic=topic, keyword=keyword, provider=provider, usage=_failed_usage(provider),
            call_count=getattr(provider, "call_count", 0), elapsed_seconds=time.perf_counter() - started,
            seo_report=None, complete=False, incomplete_reasons=["Section rewrite failed."],
            cost=compute_cost(provider.provider_id, provider.model, "unavailable", 0, 0),
            status="failed", error=str(exc), run_id=run_id,
            artifacts={"parent_slug": record["slug"], "kind": "section_revision"},
        )
        save_run_metrics(failed, metrics_dir)
        raise
    elapsed = time.perf_counter() - started

    article.sections = [(h, body if i == section_index else b) for i, (h, b) in enumerate(article.sections)]
    article.revision = next_revision(output_dir, base_slug)
    article.run_id = run_id
    target = getattr(provider, "target_words", None)
    checks = run_checks(article)
    seo = run_seo_checks(article, target_words=target)  # recalculated for the new text
    complete, reasons = check_completeness(
        article, create_outline(topic, keyword), completeness_target(provider, target)
    )
    path, report_path = save_draft(article, output_dir, seo, filename=f"{base_slug}-rev{article.revision}.md")

    tracker = TokenTracker()
    _usage_for(provider, tracker, topic, keyword, [], article)
    usage = tracker.summary()
    cost = compute_cost(
        provider.provider_id, provider.model, usage.usage_source, usage.actual_input_tokens or 0,
        usage.actual_output_tokens or 0, usage.cached_input_tokens, pricing,
    ) if usage.usage_source == "actual" else compute_cost(
        provider.provider_id, provider.model, usage.usage_source, 0, 0, None, pricing)
    markdown = path.read_text(encoding="utf-8")
    metrics = build_run_metrics(
        topic=topic, keyword=keyword, provider=provider, usage=usage,
        call_count=getattr(provider, "call_count", 0), elapsed_seconds=elapsed, seo_report=seo,
        complete=complete, incomplete_reasons=reasons, cost=cost, regenerated_sections=[heading],
        run_id=run_id,
        artifacts={"article_slug": path.stem, "revision": article.revision, "draft_path": str(path),
                   "seo_report_path": str(report_path), "markdown_sha256": text_hash(markdown),
                   "parent_slug": record["slug"], "kind": "section_revision"},
    )
    save_run_metrics(metrics, metrics_dir)
    return new_record(
        path.stem, article, checks, seo, metrics, record["length"], markdown, base_slug=base_slug,
        parent_slug=record["slug"], note=f"Section '{heading}' rewritten (1 model call).",
    )
