"""Content-quality report and status for one article revision.

Results stay separate on purpose (SEO structure, readability, keyword use, completeness, sources,
human review, export readiness). There is no single "ready" score. A status is only reported when
its requirement is really met; the two approval statuses can only come from a person clicking
the matching button for this exact revision.
"""

SOURCE_LABELS = {
    "not_performed": "Source verification NOT performed.",
    "no_sources": "No sources are listed, so no claim has a checked source.",
    "incomplete": "Some listed sources could not be reached or have problems.",
    "links_reachable": "Listed links are reachable. Claims are still NOT verified.",
}

STATUS_LABELS = {
    "draft_generated": "Draft generated",
    "automated_checks_completed": "Automated checks completed",
    "source_verification_incomplete": "Source verification incomplete",
    "needs_human_review": "Needs human review",
    "approved_for_export": "Approved for export",
    "approved_for_publication": "Approved for publication by a human",
}


def source_state(sources):
    """'not_performed' unless a check was actually run for this revision."""
    return sources.get("overall", "incomplete") if sources and sources.get("performed") else "not_performed"


def _split_seo(messages):
    keyword = [m for m in messages if "keyword" in m.lower()]
    reading = [m for m in messages if "sentence" in m.lower()]
    structure = [m for m in messages if m not in keyword and m not in reading]
    return structure, keyword, reading


def build_quality_report(data, stale=False, approvals=None):
    """`data` is the saved web record of one revision. Returns separate sections and a status."""
    seo, metrics = data["seo"], data["metrics"]
    approvals = approvals if approvals is not None else data.get("approvals") or {}
    src_state = source_state(data.get("sources"))

    sections = {}
    parts = {lvl: _split_seo(seo[lvl]) for lvl in ("failed", "warnings", "passed")}
    for index, name in enumerate(("seo_structure", "keyword_use", "readability")):
        sections[name] = {lvl: parts[lvl][index] for lvl in ("failed", "warnings", "passed")}
    sections["readability"]["stats"] = seo.get("readability", {})
    sections["completeness"] = {
        "complete": bool(metrics.get("appears_complete")),
        "reasons": metrics.get("incomplete_reasons", []),
    }
    sections["sources"] = {
        "state": src_state,
        "summary": SOURCE_LABELS[src_state],
        "details": (data.get("sources") or {}).get("sources", []),
        "limits": "Source checks are separate from the SEO checklist and never mean medical validation.",
    }
    flags = list(data["article"].get("review_flags", [])) + [
        f"Needs evidence: {c}" for c in seo.get("claims_needing_evidence", [])
    ]
    sections["human_review"] = {
        "required": True,
        "flags": flags,
        "message": "A qualified person must check facts, sources and tone before anything is published.",
    }

    blockers, notes = [], []
    if not sections["completeness"]["complete"]:
        blockers.append("The article looks incomplete: " + " ".join(sections["completeness"]["reasons"]))
    if data.get("legacy"):
        notes.append("Created before run/report tracking: the checks cannot be proven to match this file.")
    if stale:
        notes.append("The saved report no longer matches the article file (it changed after the checks ran).")
    if seo["failed"]:
        notes.append(f"{len(seo['failed'])} SEO check(s) failed.")
    if src_state != "links_reachable":
        notes.append(SOURCE_LABELS[src_state])
    sections["export_readiness"] = {"allowed": not blockers, "blockers": blockers, "warnings": notes}

    fresh = not stale
    reached = {
        "draft_generated": True,
        "automated_checks_completed": fresh,
        "source_verification_incomplete": fresh and src_state != "links_reachable",
        "needs_human_review": fresh and not approvals.get("publication"),
        "approved_for_export": fresh and bool(approvals.get("export")),
        "approved_for_publication": fresh and bool(approvals.get("publication")),
    }
    order = list(STATUS_LABELS)
    if reached["approved_for_publication"]:
        current = "approved_for_publication"
    elif reached["approved_for_export"]:
        current = "approved_for_export"
    elif reached["source_verification_incomplete"]:
        current = "source_verification_incomplete"
    elif reached["needs_human_review"]:
        current = "needs_human_review"
    elif reached["automated_checks_completed"]:
        current = "automated_checks_completed"
    else:
        current = "draft_generated"
    return {
        "sections": sections,
        "status": {
            "current": current,
            "label": STATUS_LABELS[current],
            "reached": {k: reached[k] for k in order},
            "publication_ready": reached["approved_for_publication"],
        },
    }


def can_approve_export(data, stale):
    """Return (allowed, reason). Only a complete, fresh revision can be approved for export."""
    if data.get("legacy"):
        return False, "This draft was created before report tracking existed, so its checks cannot be tied to the file."
    if stale:
        return False, "The report is out of date for this article file."
    if not data["metrics"].get("appears_complete"):
        return False, "The article looks incomplete."
    return True, ""
