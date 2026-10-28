"""Per-run metrics. Saved as JSON with unique names; earlier runs are never overwritten.

No API keys and no full prompts are stored.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.seo_checks import count_words  # noqa: E402

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "reports" / "runs"


def check_completeness(article, outline, target_words=None):
    """Return (complete, reasons). Heuristic: it cannot prove an article is good."""
    reasons = []
    if len(article.sections) < len(outline):
        reasons.append(f"Only {len(article.sections)} of {len(outline)} planned sections.")
    if any(not body.strip() for _, body in article.sections):
        reasons.append("A section has no body text.")
    if not article.faq:
        reasons.append("FAQ is missing.")
    elif any(not a.strip() for _, a in article.faq):
        reasons.append("An FAQ answer is empty.")
    if not article.title.strip() or not article.meta_description.strip():
        reasons.append("Title or meta description is missing.")
    if target_words:
        words = count_words(article)  # same count as the SEO report: sections + FAQ
        if words < 0.4 * target_words:
            reasons.append(f"Only about {words} words for a {target_words}-word target.")
    last = article.sections[-1][1].strip() if article.sections else ""
    if last and last[-1] not in ".!?)\"'":
        reasons.append("The last section does not end with sentence punctuation (possible cut-off).")
    return (not reasons), reasons


def _reasoning_total(provider):
    values = [c.get("reasoning_tokens") for c in getattr(provider, "call_log", [])]
    values = [v for v in values if v is not None]
    return sum(values) if values else None


def build_run_metrics(*, topic, keyword, provider, usage, call_count, elapsed_seconds,
                      seo_report, complete, incomplete_reasons, cost, status="ok",
                      regenerated_sections=None, error=None, run_id=None, artifacts=None, extra=None):
    in_tokens = out_tokens = None
    if usage is not None:
        if usage.usage_source == "actual":
            in_tokens, out_tokens = usage.actual_input_tokens, usage.actual_output_tokens
        elif usage.usage_source == "estimated":
            in_tokens, out_tokens = usage.estimated_input_tokens, usage.estimated_output_tokens
    metrics = {
        "run_id": run_id or uuid.uuid4().hex[:8],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "error": error,
        "topic": topic,
        "keyword": keyword,
        "provider": getattr(provider, "provider_id", "unknown"),
        "model": getattr(provider, "model", None),
        "prompt_style": getattr(provider, "prompt_style", None),
        "target_words": getattr(provider, "target_words", None),
        "usage_source": usage.usage_source if usage else "unavailable",
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cached_input_tokens": usage.cached_input_tokens if usage else None,
        "reasoning_tokens": _reasoning_total(provider),
        "model_calls": call_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "cost": cost,
        "regenerated_sections": regenerated_sections or [],
        "artifacts": artifacts or {},
    }
    if seo_report is not None:
        metrics.update(
            word_count=seo_report.word_count,
            seo={
                "passed": len(seo_report.passed),
                "warnings": len(seo_report.warnings),
                "failed": len(seo_report.failed),
                "failed_items": list(seo_report.failed),
                "warning_items": list(seo_report.warnings),
            },
            appears_complete=complete,
            incomplete_reasons=incomplete_reasons,
            human_review_flags={
                "needs_human_review": seo_report.needs_human_review,
                "claims_needing_evidence": len(seo_report.claims_needing_evidence),
                "source_references_found": len(seo_report.references_found),
                "source_verification": seo_report.source_verification,
            },
        )
    else:
        metrics.update(word_count=None, seo=None, appears_complete=False,
                       incomplete_reasons=incomplete_reasons, human_review_flags=None)
    metrics.update(extra or {})
    return metrics


def save_run_metrics(metrics, runs_dir=DEFAULT_RUNS_DIR):
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", str(metrics["topic"]).lower()).strip("-")[:40] or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = runs_dir / f"run-{stamp}-{metrics['provider']}-{slug}-{metrics['run_id']}.json"
    with open(path, "x", encoding="utf-8") as handle:  # "x" refuses to overwrite
        json.dump(metrics, handle, indent=2)
    return path
