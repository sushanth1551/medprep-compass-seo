"""Quality workflow tests: structured output, SEO checks, sources, status, revisions, consistency, export.

Everything uses stubs or the mock provider: no network, no credits, no real key.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.web as web
from app import store
from app.config import output_limit_warning
from app.models import Article, ProviderError
from app.providers.groq_provider import GroqProvider
from app.providers.prompts import ARTICLE_LENGTHS, STRUCTURED_SYSTEM, system_prompt
from app.quality import build_quality_report
from app.seo_checks import SeoConfig, run_seo_checks
from app.sources import FetchResult, _is_public_host, check_sources, claim_overlap, default_fetch
from app.workflow import run_workflow

KEY = "gsk-quality-test-not-real"


def full_reply(**over):
    data = {
        "title": "How to Plan USMLE Step 1 Study Time Well",
        "meta_description": "A practical usmle step 1 study plan guide with weekly structure and common mistakes to avoid.",
        "search_intent": "Informational: students planning Step 1 study time",
        "audience": "Medical students",
        "sections": [{"heading": h, "body": f"{h} text. " * 30 + "It ends cleanly."}
                     for h in ("Introduction", "Key Concepts to Know", "A Practical Study Plan",
                               "Common Mistakes to Avoid", "Conclusion")],
        "faq": [{"question": "Q one?", "answer": "A reasonably long answer that explains the point in some detail here."}],
        "references": [{"title": "USMLE Step 1", "url": "https://www.usmle.org/step-exams/step-1",
                        "claim": "Step 1 assesses sciences basic to the practice of medicine"}],
        "review_flags": ["Check the current exam format on usmle.org"],
    }
    data.update(over)
    return json.dumps(data)


def stub(text, finish="stop"):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=900, prompt_tokens_details=None,
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=300)))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), calls=calls)


# Structured generation ---------------------------------------------------------------------------
def test_structured_fields_flow_into_the_article_and_markdown(tmp_path):
    provider = GroqProvider(KEY, "m", client=stub(full_reply()), target_words=900)
    result = run_workflow("Plan Step 1", "usmle step 1 study plan", provider, tmp_path)
    article = result.article
    assert article.search_intent.startswith("Informational") and article.audience == "Medical students"
    assert article.references[0]["url"] == "https://www.usmle.org/step-exams/step-1"
    assert article.review_flags == ["Check the current exam format on usmle.org"]
    md = open(result.output_path, encoding="utf-8").read()
    assert "**Search intent:**" in md and "## References" in md and "https://www.usmle.org/step-exams/step-1" in md
    assert f"**Run ID:** {article.run_id} | **Revision:** 1" in md
    assert result.metrics["run_id"] == article.run_id


def test_structured_prompt_is_default_and_forbids_invention():
    assert system_prompt("structured") == STRUCTURED_SYSTEM
    for needle in ("Never invent", "references", "review_flags", "search_intent"):
        assert needle in STRUCTURED_SYSTEM
    assert GroqProvider(KEY, "m", client=stub(full_reply())).prompt_style == "structured"


def test_malformed_optional_fields_do_not_fail_the_article(tmp_path):
    reply = full_reply(references="not a list", review_flags=None, audience=None)
    result = run_workflow("Plan", provider=GroqProvider(KEY, "m", client=stub(reply)), output_dir=tmp_path)
    assert result.article.references == [] and result.article.audience == ""


def test_older_replies_without_new_fields_still_work(tmp_path):
    minimal = json.dumps({"title": "T", "meta_description": "M", "faq": [{"question": "Q", "answer": "A"}],
                          "sections": [{"heading": h, "body": "B."} for h in ("H1", "H2", "H3")]})
    result = run_workflow("Plan", provider=GroqProvider(KEY, "m", client=stub(minimal)), output_dir=tmp_path)
    assert result.article.references == []


def test_length_choices_match_the_requested_targets():
    assert ARTICLE_LENGTHS["short"][0] in range(500, 601)
    assert ARTICLE_LENGTHS["medium"][0] == 900 and ARTICLE_LENGTHS["long"][0] == 1500
    client = stub(full_reply())
    GroqProvider(KEY, "m", client=client, target_words=1500).generate_sections("t", "k", [])
    assert "about 1500 words" in client.calls[0]["messages"][1]["content"]


def test_output_limit_warning_only_when_limit_is_too_small():
    assert output_limit_warning(8000, 1500) is None
    assert "too low" in output_limit_warning(2000, 1500)


# SEO checks ------------------------------------------------------------------------------------------
def article(**over):
    base = dict(
        title="How to Plan USMLE Step 1 Study Time Well", keyword="usmle step 1",
        meta_description="A practical usmle step 1 guide with weekly structure and common mistakes to avoid today.",
        sections=[("One", "Plan your week well. " * 40), ("Two", "Review daily and often. " * 40),
                  ("Three", "Practice questions steadily. " * 40)],
        faq=[("Q?", "This answer has enough words to be useful to a reader who asks it.")],
    )
    base.update(over)
    return Article(**base)


def test_title_and_meta_length_checks():
    report = run_seo_checks(article(title="Short", meta_description="Too short usmle step 1"))
    text = " ".join(report.warnings)
    assert "Title is 5 characters" in text and "Meta description is only" in text
    ok = run_seo_checks(article())
    assert any("Title length" in p for p in ok.passed)


def test_title_length_thresholds_are_configurable():
    report = run_seo_checks(article(title="Short"), config=SeoConfig(title_min_chars=3))
    assert not any("Title is 5" in w for w in report.warnings)


def test_heading_hierarchy_failures_and_passes():
    bad = article(sections=[("One", "x y z.\n\n#### Deep"), ("Two", "a b."), ("Three", "c d.")])
    assert any("H3" in f for f in run_seo_checks(bad).failed)
    good = article(sections=[("One", "x y z.\n\n### Sub heading\n\nmore."), ("Two", "a b."), ("Three", "c d.")])
    report = run_seo_checks(good)
    assert not any("H3" in f for f in report.failed)
    assert any("H3 under H2" in p for p in report.passed)


def test_thin_faq_and_length_against_target():
    report = run_seo_checks(article(faq=[("Q?", "Short.")]), target_words=900)
    assert any("FAQ answer" in w for w in report.warnings)
    assert any("floor for a 900-word target" in w for w in report.warnings)
    near = run_seo_checks(article(), target_words=500)
    assert any("near the 500-word target" in p for p in near.passed) or any("ceiling" in w for w in near.warnings)


def test_bad_reference_urls_fail_and_good_ones_pass_syntax_only():
    bad = run_seo_checks(article(references=[{"title": "x", "url": "ftp://nope", "claim": ""}]))
    assert any("valid web address" in f for f in bad.failed)
    good = run_seo_checks(article(references=[{"title": "x", "url": "https://www.usmle.org/", "claim": ""}]))
    assert any("existence NOT checked" in p for p in good.passed)


# Source verification --------------------------------------------------------------------------------------
REFS = [{"title": "USMLE", "url": "https://www.usmle.org/step-exams/step-1",
         "claim": "Step 1 assesses sciences basic to the practice of medicine"}]


def fake_fetch(pages):
    return lambda url: pages.get(url, FetchResult(error="Timed out."))


def test_sources_not_checked_means_not_performed():
    data = {"article": {"review_flags": []}, "seo": run_seo_checks(article()).__dict__ | {"readability": {}},
            "metrics": {"appears_complete": True}, "sources": None, "approvals": {}}
    report = build_quality_report(data)
    assert report["sections"]["sources"]["state"] == "not_performed"
    assert "NOT performed" in report["sections"]["sources"]["summary"]
    assert report["status"]["label"] == "Source verification incomplete"


def test_reachable_source_with_matching_words_is_a_hint_not_verification():
    page = FetchResult(200, REFS[0]["url"], "<html>The sciences basic to the practice of medicine. Step 1 assesses</html>")
    result = check_sources(article(references=REFS), fetch=fake_fetch({REFS[0]["url"]: page}))
    src = result["sources"][0]
    assert result["overall"] == "links_reachable" and src["reachable"] and src["support"] == "words_found"
    assert "not proof" in src["note"] and "verified" not in result["overall"]


def test_unreachable_and_invalid_sources_are_flagged():
    refs = REFS + [{"title": "bad", "url": "not a url", "claim": ""}]
    result = check_sources(article(references=refs), fetch=fake_fetch({}))
    assert result["overall"] == "incomplete"
    assert [s["reachable"] for s in result["sources"]] == [False, False]
    assert result["sources"][1]["note"] == "Not a valid web address."


def test_article_without_sources_says_so_and_missing_words_warn():
    assert check_sources(article(), fetch=fake_fetch({}))["overall"] == "no_sources"
    page = FetchResult(200, "u", "<p>Completely unrelated cooking recipes and gardening tips.</p>")
    miss = check_sources(article(references=REFS), fetch=fake_fetch({REFS[0]["url"]: page}))
    assert miss["sources"][0]["support"] == "words_missing"


def test_claim_overlap_and_private_address_protection():
    assert claim_overlap("tiny", "text") is None
    assert claim_overlap("alpha bravo charlie delta", "<b>alpha bravo charlie</b>") == 0.75
    assert _is_public_host("127.0.0.1") is False
    assert default_fetch("http://127.0.0.1/secret").error  # refused without any network call


# Status ---------------------------------------------------------------------------------------------------
def record_for(tmp_path, sources=None, approvals=None, complete=True):
    result = run_workflow("Plan", provider=GroqProvider(KEY, "m", client=stub(full_reply())), output_dir=tmp_path,
                          metrics_dir=tmp_path / "runs")
    md = open(result.output_path, encoding="utf-8").read()
    rec = store.new_record(result.article and "plan", result.article, result.check_results, result.seo_report,
                           result.metrics, "medium", md)
    rec["sources"], rec["approvals"] = sources, approvals or {}
    rec["metrics"]["appears_complete"] = complete
    return rec


def test_status_only_advances_when_requirements_are_met(tmp_path):
    rec = record_for(tmp_path)
    assert build_quality_report(rec)["status"]["current"] == "source_verification_incomplete"
    reachable = {"performed": True, "overall": "links_reachable", "sources": []}
    rec["sources"] = reachable
    assert build_quality_report(rec)["status"]["current"] == "needs_human_review"
    rec["approvals"] = {"export": {"at": "now"}}
    assert build_quality_report(rec)["status"]["current"] == "approved_for_export"
    rec["approvals"]["publication"] = {"at": "now"}
    status = build_quality_report(rec)["status"]
    assert status["current"] == "approved_for_publication" and status["publication_ready"]
    stale = build_quality_report(rec, stale=True)["status"]
    assert stale["current"] == "draft_generated" and not stale["publication_ready"]


def test_report_sections_are_separate_and_incomplete_blocks_export(tmp_path):
    report = build_quality_report(record_for(tmp_path, complete=False))
    assert set(report["sections"]) >= {"seo_structure", "keyword_use", "readability", "completeness", "sources",
                                       "human_review", "export_readiness"}
    assert report["sections"]["export_readiness"]["allowed"] is False
    assert report["sections"]["human_review"]["required"] is True


# Consistency -----------------------------------------------------------------------------------------------
def test_consistency_detects_stale_and_mismatched_reports(tmp_path):
    result = run_workflow("Plan", provider=GroqProvider(KEY, "m", client=stub(full_reply())), output_dir=tmp_path,
                          metrics_dir=tmp_path / "runs")
    slug = result.output_path.rsplit("\\", 1)[-1][:-3] if "\\" in result.output_path else result.output_path.rsplit("/", 1)[-1][:-3]
    md = open(result.output_path, encoding="utf-8").read()
    rec = store.new_record(slug, result.article, result.check_results, result.seo_report, result.metrics, "medium", md)
    assert store.consistency_problems(tmp_path, rec) == []
    # 1. the article is edited after its checks ran
    open(result.output_path, "a", encoding="utf-8").write("\nEdited later.")
    problems = store.consistency_problems(tmp_path, rec)
    assert store.is_stale(problems) and any("stale" in p for p in problems)
    # 2. metrics from another run
    rec2 = {**rec, "markdown_sha256": store.text_hash(open(result.output_path, encoding="utf-8").read())}
    rec2["metrics"] = {**rec["metrics"], "run_id": "other000"}
    assert any("different run" in p for p in store.consistency_problems(tmp_path, rec2))


def test_report_file_carries_run_id_revision_and_hash(tmp_path):
    result = run_workflow("Plan", provider=GroqProvider(KEY, "m", client=stub(full_reply())), output_dir=tmp_path)
    saved = json.load(open(result.report_path, encoding="utf-8"))
    assert saved["run_id"] == result.article.run_id and saved["revision"] == 1
    assert saved["markdown_sha256"] == store.text_hash(open(result.output_path, encoding="utf-8").read())


# Web: sources, revisions, approvals, export guard -----------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(web, "METRICS_DIR", tmp_path / "runs")
    monkeypatch.setenv("AI_PROVIDER", "mock")
    return TestClient(web.app, follow_redirects=False)


def make(client, topic="Quality flow topic"):
    response = client.post("/generate", data={"topic": topic, "keyword": "", "length": "short"})
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[1]


def test_health_endpoint_reveals_nothing(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_page_shows_status_sections_and_source_warning(client):
    slug = make(client)
    page = client.get(f"/articles/{slug}").text
    for needle in ("Status:", "Quality report", "Completeness", "SEO structure", "Sources and facts",
                   "Source verification NOT performed", "Human review required", "Run ID",
                   "Check source links", "Download PDF", "Rewrite this section"):
        assert needle in page
    assert "Approved for publication" not in page.split("Your decisions")[0].split("Status:")[1][:200]


def test_check_sources_route_records_result_without_claiming_verification(client, monkeypatch):
    slug = make(client)
    monkeypatch.setattr(web, "check_sources", lambda art: {"performed": True, "overall": "links_reachable",
                        "sources": [], "checked_at": "now", "limits": "x"})
    assert client.post(f"/articles/{slug}/check-sources").status_code == 303
    page = client.get(f"/articles/{slug}").text
    assert "Claims are still NOT verified" in page and "Needs human review" in page


def test_revision_creates_new_file_keeps_original_and_records_calls(client):
    slug = make(client)
    original = (web.OUTPUT_DIR / f"{slug}.md").read_text(encoding="utf-8")
    response = client.post(f"/articles/{slug}/revise", data={"section": 1})
    assert response.status_code == 303
    new_slug = response.headers["location"].rsplit("/", 1)[1]
    assert new_slug == f"{slug}-rev2"
    assert (web.OUTPUT_DIR / f"{slug}.md").read_text(encoding="utf-8") == original  # untouched
    new_md = (web.OUTPUT_DIR / f"{new_slug}.md").read_text(encoding="utf-8")
    assert "Rewritten section" in new_md and "**Revision:** 2" in new_md
    old_rec, new_rec = store.load_record(web.OUTPUT_DIR, slug), store.load_record(web.OUTPUT_DIR, new_slug)
    assert new_rec["parent_slug"] == slug and new_rec["revision"] == 2
    assert new_rec["run_id"] != old_rec["run_id"] and new_rec["metrics"]["run_id"] == new_rec["run_id"]
    assert new_rec["sources"] is None and new_rec["approvals"] == {}  # old checks do not carry over
    assert store.consistency_problems(web.OUTPUT_DIR, new_rec) == []
    assert len(list(web.METRICS_DIR.glob("run-*.json"))) == 2
    assert "r2" in client.get(f"/articles/{new_slug}").text


def test_revision_validation_and_real_provider_call_counting(client, monkeypatch):
    slug = make(client)
    assert client.post(f"/articles/{slug}/revise", data={"section": 99}).status_code == 400
    calls = stub(json.dumps({"body": "A rewritten section body."}))
    monkeypatch.setattr(web, "build_provider", lambda **kw: GroqProvider(KEY, "m", client=calls, **kw))
    response = client.post(f"/articles/{slug}/revise", data={"section": 0})
    assert response.status_code == 303 and len(calls.calls) == 1  # exactly one visible paid call
    metrics = store.load_record(web.OUTPUT_DIR, f"{slug}-rev2")["metrics"]
    assert metrics["model_calls"] == 1 and metrics["usage_source"] == "actual"
    assert metrics["input_tokens"] == 200 and metrics["output_tokens"] == 900


def test_failed_revision_changes_nothing_and_is_recorded(client, monkeypatch):
    slug = make(client)
    monkeypatch.setattr(web, "build_provider", lambda **kw: GroqProvider(
        KEY, "m", client=stub("{broken", finish="length"), **kw))
    response = client.post(f"/articles/{slug}/revise", data={"section": 0})
    assert response.status_code == 502 and "cut off" in response.text
    assert not (web.OUTPUT_DIR / f"{slug}-rev2.md").exists()
    assert any(json.loads(p.read_text(encoding="utf-8"))["status"] == "failed" for p in web.METRICS_DIR.iterdir())


def test_approval_flow_requires_each_step_and_confirmation(client):
    slug = make(client)
    assert client.post(f"/articles/{slug}/approve", data={"level": "publication", "confirm": "yes"}).status_code == 409
    assert client.post(f"/articles/{slug}/approve", data={"level": "export"}).status_code == 303
    assert client.post(f"/articles/{slug}/approve", data={"level": "publication"}).status_code == 400
    before = client.get(f"/articles/{slug}").text
    assert "Approved for export" in before and "never publishes anything" in before
    assert client.post(f"/articles/{slug}/approve", data={"level": "publication", "confirm": "yes"}).status_code == 303
    page = client.get(f"/articles/{slug}").text
    assert "Approved for publication by a human" in page


def test_stale_article_shows_warning_and_blocks_approval(client):
    slug = make(client)
    with open(web.OUTPUT_DIR / f"{slug}.md", "a", encoding="utf-8") as handle:
        handle.write("\nEdited outside the app.")
    page = client.get(f"/articles/{slug}").text
    assert "Consistency problems" in page and "stale" in page
    assert client.post(f"/articles/{slug}/approve", data={"level": "export"}).status_code == 409


def test_incomplete_article_cannot_be_exported_or_approved(client, monkeypatch):
    tiny = [{"heading": h, "body": "Far too short."} for h in ("A", "B", "C", "D", "E")]
    short = full_reply(sections=tiny)
    monkeypatch.setattr(web, "build_provider", lambda **kw: GroqProvider(KEY, "m", client=stub(short), **kw))
    response = client.post("/generate", data={"topic": "Incomplete one", "keyword": "", "length": "medium"})
    slug = response.headers["location"].rsplit("/", 1)[1]
    page = client.get(f"/articles/{slug}").text
    assert "Export is blocked" in page and "Incomplete:" in page
    assert client.get(f"/articles/{slug}/download").status_code == 409
    assert client.get(f"/articles/{slug}/download.pdf").status_code == 409
    assert client.post(f"/articles/{slug}/approve", data={"level": "export"}).status_code == 409


def test_markdown_and_pdf_exports_match_the_displayed_revision(client):
    from io import BytesIO

    from pypdf import PdfReader

    slug = make(client)
    new_slug = client.post(f"/articles/{slug}/revise", data={"section": 2}).headers["location"].rsplit("/", 1)[1]
    record = store.load_record(web.OUTPUT_DIR, new_slug)
    md = client.get(f"/articles/{new_slug}/download")
    pdf = client.get(f"/articles/{new_slug}/download.pdf")
    assert md.headers["x-article-run-id"] == pdf.headers["x-article-run-id"] == record["run_id"]
    assert md.headers["x-article-revision"] == pdf.headers["x-article-revision"] == "2"
    on_disk = (web.OUTPUT_DIR / f"{new_slug}.md").read_text(encoding="utf-8")
    assert md.content.decode("utf-8").splitlines() == on_disk.splitlines()  # same text, any line endings
    text = "\n".join(p.extract_text() for p in PdfReader(BytesIO(pdf.content)).pages)
    assert record["run_id"] in text and "Revision: 2" in text.replace("\n", " ")
    assert "Rewritten section" in text


def test_body_rendering_escapes_html_and_formats_markdown():
    html = str(web.render_body("Text <script>x</script> **bold**\n\n### Sub\n\n- a\n- b\n\n1. one"))
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<strong>bold</strong>" in html and "<h4>Sub</h4>" in html and "<ul><li>a</li>" in html and "<ol>" in html


def test_historical_files_and_secrets_survive_a_full_flow(client, monkeypatch, tmp_path):
    web.OUTPUT_DIR.mkdir()
    old = web.OUTPUT_DIR / "historic-draft.md"
    old.write_text("# Historic\n\ntext", encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    slug = make(client)
    client.post(f"/articles/{slug}/revise", data={"section": 0})
    assert old.read_text(encoding="utf-8") == "# Historic\n\ntext"
    blob = "".join(p.read_text(encoding="utf-8") for p in list(web.OUTPUT_DIR.iterdir()) + list(web.METRICS_DIR.iterdir()))
    assert KEY not in blob
    assert ProviderError  # imported for the failure tests above


def test_legacy_sidecar_from_an_earlier_version_loads_and_cannot_be_approved(client):
    """A real-world case: records written before run ids, hashes and revisions existed."""
    web.OUTPUT_DIR.mkdir()
    (web.OUTPUT_DIR / "old-web.md").write_text("# Old web draft\n\ntext", encoding="utf-8")
    legacy = {
        "slug": "old-web", "created_utc": "2026-01-01T00:00:00+00:00", "length": "medium",
        "article": {"title": "Old web draft", "meta_description": "m", "keyword": "k",
                    "sections": [["Intro", "text."]], "faq": [["Q", "A"]], "notice": "n", "provider_name": "p"},
        "checks": {}, "seo": {"passed": [], "warnings": [], "failed": [], "word_count": 3, "readability": {},
                              "claims_needing_evidence": [], "references_found": [], "score_text": "0/0",
                              "source_verification": "NOT PERFORMED"},
        "metrics": {"run_id": "abc12345", "provider": "groq", "model": "m", "usage_source": "actual",
                    "input_tokens": 1, "output_tokens": 2, "model_calls": 1, "elapsed_seconds": 1.0,
                    "appears_complete": True, "incomplete_reasons": [],
                    "cost": {"amount": None, "currency": "USD", "note": "unknown"}},
    }
    (web.OUTPUT_DIR / "old-web.web.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert client.get("/").status_code == 200
    page = client.get("/articles/old-web")
    assert page.status_code == 200 and "earlier version of the app" in page.text
    assert client.get("/articles/old-web/download.pdf").status_code == 200
    assert client.post("/articles/old-web/approve", data={"level": "export"}).status_code == 409


# Found by the live run: negated "guarantee" and heading levels ----------------------------------------------
@pytest.mark.parametrize("text, promised", [
    ("The goal is not to guarantee a perfect score.", False),
    ("No plan can guarantee a pass, but this helps.", False),
    ("We cannot guarantee results.", False),
    ("This plan will guarantee you pass.", True),
    ("Follow this and you will pass the exam.", True),
    ("It is a hard exam. We guarantee success.", True),  # negation in a previous sentence must not count
])
def test_exam_promise_check_ignores_negated_statements(text, promised):
    report = run_seo_checks(article(sections=[("One", text), ("Two", "a b."), ("Three", "c d.")]))
    assert any("promise" in f for f in report.failed) is promised


def test_heading_levels_may_go_one_deeper_but_not_skip():
    ok = article(sections=[("One", "x.\n\n### A\n\ny.\n\n#### B\n\nz."), ("Two", "a b."), ("Three", "c d.")])
    assert not [f for f in run_seo_checks(ok).failed if "Heading" in f]
    skip = article(sections=[("One", "x.\n\n#### Straight to H4"), ("Two", "a b."), ("Three", "c d.")])
    assert any("found H4 after H2" in f for f in run_seo_checks(skip).failed)
    h1 = article(sections=[("One", "# Another title"), ("Two", "a b."), ("Three", "c d.")])
    assert any("Heading levels" in f for f in run_seo_checks(h1).failed)


def test_recheck_route_updates_the_checks_but_not_the_article(client):
    slug = make(client)
    before = (web.OUTPUT_DIR / f"{slug}.md").read_bytes()
    record = store.load_record(web.OUTPUT_DIR, slug)
    record["seo"]["failed"] = ["Stale failure from an older version of the checks."]
    store.save_record(web.OUTPUT_DIR, record)
    assert "Stale failure" in client.get(f"/articles/{slug}").text
    assert client.post(f"/articles/{slug}/recheck").status_code == 303
    page = client.get(f"/articles/{slug}").text
    assert "Stale failure" not in page and "re-run" in page.lower()
    assert (web.OUTPUT_DIR / f"{slug}.md").read_bytes() == before
    assert store.consistency_problems(web.OUTPUT_DIR, store.load_record(web.OUTPUT_DIR, slug)) == []


def test_reply_with_most_sections_missing_is_a_failed_generation_not_a_saved_draft(client, monkeypatch):
    """Found in real use: the model finished normally but returned 1 of 5 sections."""
    one = full_reply(sections=[{"heading": "Conclusion", "body": "Only the ending."}])
    monkeypatch.setattr(web, "build_provider", lambda **kw: GroqProvider(KEY, "m", client=stub(one), **kw))
    response = client.post("/generate", data={"topic": "Missing sections", "keyword": "", "length": "medium"})
    assert response.status_code == 502
    assert "only 1 of 5 requested sections" in response.text and "not a token-limit problem" in response.text
    assert "follow-up request" in response.text  # exactly one bounded attempt to fill the gap was made
    assert not list(web.OUTPUT_DIR.glob("*.md")) if web.OUTPUT_DIR.exists() else True
    failed = json.loads(next(web.METRICS_DIR.glob("run-*.json")).read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["usage_source"] == "actual" and failed["model_calls"] == 2


def test_blocked_export_offers_a_prefilled_new_draft_link_and_the_form_prefills(client, monkeypatch):
    tiny = [{"heading": h, "body": "Far too short."} for h in ("A", "B", "C", "D", "E")]
    monkeypatch.setattr(web, "build_provider",
                        lambda **kw: GroqProvider(KEY, "m", client=stub(full_reply(sections=tiny)), **kw))
    slug = client.post("/generate", data={"topic": "Prefill me", "keyword": "kw one", "length": "long"})
    slug = slug.headers["location"].rsplit("/", 1)[1]
    page = client.get(f"/articles/{slug}").text
    assert "Start a new draft with the same topic" in page and "topic=Prefill%20me" in page
    form = client.get("/?topic=Prefill%20me&keyword=kw%20one&length=long").text
    assert 'value="Prefill me"' in form and 'value="kw one"' in form and 'value="long" selected' in form
