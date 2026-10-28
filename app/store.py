"""File-based records for the web app. One JSON "record" sits beside each article revision.

The Markdown file is never edited after it is written; the record holds the checks, metrics,
source results and approvals for exactly that file. A hash and the run id let us notice a
report that belongs to a different run or to an older/edited version of the article.
"""

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.models import Article
from app.workflow import text_hash

SLUG_PATTERN = re.compile(r"^[a-z0-9-]+$")


def md_path(output_dir, slug):
    return Path(output_dir) / f"{slug}.md"


def record_path(output_dir, slug):
    return Path(output_dir) / f"{slug}.web.json"


def new_record(slug, article, checks, seo_report, metrics, length, markdown, base_slug=None,
               parent_slug=None, note=""):
    """Build the record for one saved revision."""
    return {
        "slug": slug,
        "base_slug": base_slug or slug,
        "parent_slug": parent_slug,
        "revision": article.revision,
        "revision_note": note,
        "run_id": article.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "length": length,
        "markdown_sha256": text_hash(markdown),
        "article": {
            "title": article.title,
            "meta_description": article.meta_description,
            "keyword": article.keyword,
            "sections": article.sections,
            "faq": article.faq,
            "notice": article.notice,
            "provider_name": article.provider_name,
            "search_intent": article.search_intent,
            "audience": article.audience,
            "references": article.references,
            "review_flags": article.review_flags,
        },
        "checks": checks,
        "seo": {**asdict(seo_report), "score_text": seo_report.score_text},
        "metrics": metrics,
        "sources": None,  # filled only when someone runs the source check for this revision
        "approvals": {},  # filled only by explicit button presses for this revision
    }


def article_from_record(record):
    a = record["article"]
    return Article(
        title=a["title"], meta_description=a["meta_description"], keyword=a["keyword"],
        sections=[tuple(s) for s in a["sections"]], faq=[tuple(f) for f in a["faq"]],
        provider_name=a.get("provider_name", "unknown"), notice=a.get("notice", ""),
        search_intent=a.get("search_intent", ""), audience=a.get("audience", ""),
        references=list(a.get("references", [])), review_flags=list(a.get("review_flags", [])),
        run_id=record["run_id"], revision=record["revision"],
    )


def save_record(output_dir, record):
    record_path(output_dir, record["slug"]).write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_record(output_dir, slug):
    path = record_path(output_dir, slug)
    if not path.exists():
        return None
    try:
        return _normalise_legacy(json.loads(path.read_text(encoding="utf-8")), slug)
    except (OSError, ValueError, KeyError):
        return None


def _normalise_legacy(record, slug):
    """Records saved by earlier versions lack run ids, hashes and revisions. Mark them as legacy:
    they can be viewed and exported, but their checks cannot be proven to match the file."""
    record.setdefault("slug", slug)
    if "markdown_sha256" not in record:
        record["legacy"] = True
        record.setdefault("run_id", (record.get("metrics") or {}).get("run_id", ""))
        record.setdefault("revision", 1)
    record.setdefault("base_slug", record["slug"])
    record.setdefault("parent_slug", None)
    record.setdefault("revision_note", "")
    record.setdefault("sources", None)
    record.setdefault("approvals", {})
    for key in ("review_flags", "references"):
        record["article"].setdefault(key, [])
    return record


def consistency_problems(output_dir, record):
    """List every way the article file, report, metrics and record disagree (empty = consistent)."""
    problems = []
    md = md_path(output_dir, record["slug"])
    if not md.exists():
        return ["The article file is missing."]
    if record.get("legacy"):
        return []  # no hash or run id was saved back then; nothing to compare
    text = md.read_text(encoding="utf-8")
    if text_hash(text) != record.get("markdown_sha256"):
        problems.append("The article file changed after its checks ran, so the SEO report is stale.")
    run_id = record.get("run_id")
    if record["metrics"].get("run_id") != run_id:
        problems.append("The run metrics belong to a different run than this article.")
    if f"**Run ID:** {run_id} " not in text:
        problems.append("The article file carries a different run ID than its record.")
    report_file = md.with_suffix(".seo-report.json")
    if report_file.exists():
        try:
            saved = json.loads(report_file.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
        if saved.get("run_id") != run_id or saved.get("markdown_sha256") != record.get("markdown_sha256"):
            problems.append("The saved SEO report file does not match this article revision.")
    return problems


def is_stale(problems):
    return any("stale" in p or "does not match" in p for p in problems)


def revisions_of(output_dir, base_slug):
    """All saved revisions of one article, oldest first."""
    found = []
    for path in Path(output_dir).glob("*.web.json"):
        rec = load_record(output_dir, path.name[: -len(".web.json")])
        if rec and rec.get("base_slug", rec["slug"]) == base_slug:
            found.append(rec)
    return sorted(found, key=lambda r: r["revision"])


def next_revision(output_dir, base_slug):
    return max([r["revision"] for r in revisions_of(output_dir, base_slug)] or [0]) + 1
