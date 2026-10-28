"""Regression: a normal-looking model reply produced a saved article with 1 of 5 sections (82 words).

Suspected path (reproduced below): the model's JSON repeated the "sections" key and Python's
json.loads silently keeps only the LAST copy ("Conclusion"). All tests use stubbed replies:
no network, no credits, no real key.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.web as web
from app import store
from app.config import get_max_continuations
from app.metrics import check_completeness
from app.models import ConfigError, ProviderError
from app.providers.base import merge_sections, missing_headings
from app.providers.groq_provider import GroqProvider
from app.seo_checks import count_words, run_seo_checks
from app.workflow import create_outline, run_workflow

KEY = "gsk-regression-test-not-real"
HEADINGS = ["Introduction", "Key Concepts to Know", "A Practical Study Plan", "Common Mistakes to Avoid", "Conclusion"]
BODY = "This section explains the idea in plain words. " * 20


def section(heading, body=BODY):
    return {"heading": heading, "body": body}


def reply_text(sections=None, **over):
    data = {
        "title": "How to Prepare for USMLE Step 1: A Practical Guide",
        "meta_description": "A practical usmle step 1 guide with a weekly plan and the mistakes to avoid when studying.",
        "sections": sections if sections is not None else [section(h) for h in HEADINGS],
        "faq": [{"question": "How long?", "answer": "It depends on your schedule and starting point, so plan around both."}],
    }
    data.update(over)
    return json.dumps(data)


class Replies:
    """A fake Groq client that returns the queued replies one by one and records every request."""

    def __init__(self, *replies, finish="stop"):
        self.queue = list(replies)
        self.calls = []
        self.finish = finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        finish = self.finish.pop(0) if isinstance(self.finish, list) else self.finish
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=400, prompt_tokens_details=None,
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=50)))


def provider(client, **kw):
    kw.setdefault("target_words", 900)
    return GroqProvider(KEY, "m", client=client, **kw)


# 1. normal complete article ----------------------------------------------------------------------
def test_normal_complete_article_is_saved_validated_and_exportable(tmp_path, monkeypatch):
    client = Replies(reply_text())
    result = run_workflow("Prepare for step 1", "usmle step 1", provider(client), tmp_path, metrics_dir=tmp_path / "runs")
    assert len(client.calls) == 1 and result.metrics["model_calls"] == 1
    assert [h for h, _ in result.article.sections] == HEADINGS
    assert result.metrics["appears_complete"] is True
    assert result.metrics["finish_reasons"] == ["stop"] and result.metrics["continuation_calls"] == 0
    assert result.metrics["diagnostics_file"] == ""  # nothing unusual happened


def test_complete_article_exports_through_the_web_app(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path / "o")
    monkeypatch.setattr(web, "METRICS_DIR", tmp_path / "r")
    monkeypatch.setattr(web, "build_provider", lambda **kw: provider(Replies(reply_text()), **kw))
    c = TestClient(web.app, follow_redirects=False)
    slug = c.post("/generate", data={"topic": "Prepare", "keyword": "usmle step 1", "length": "medium"}) \
        .headers["location"].rsplit("/", 1)[1]
    md, pdf = c.get(f"/articles/{slug}/download"), c.get(f"/articles/{slug}/download.pdf")
    assert md.status_code == 200 and pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert md.headers["x-article-run-id"] == pdf.headers["x-article-run-id"]


# 2. the observed bad reply ---------------------------------------------------------------------------
def test_observed_one_section_reply_is_never_saved_as_an_article(tmp_path):
    only_conclusion = reply_text([section("Conclusion", "Preparing is a marathon, not a sprint. " * 5)])
    client = Replies(only_conclusion)
    with pytest.raises(ProviderError, match="only 1 of 5 requested sections"):
        run_workflow("Prepare", provider=provider(client, max_continuations=0), output_dir=tmp_path,
                     metrics_dir=tmp_path / "runs")
    assert not list(tmp_path.glob("*.md"))  # nothing saved
    failed = json.loads(next((tmp_path / "runs").glob("run-*.json")).read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["finish_reasons"] == ["stop"]
    assert failed["diagnostics_file"] == ""  # not requested here


def test_partial_article_is_saved_only_as_incomplete_and_flagged(tmp_path):
    partial = reply_text([section(h) for h in HEADINGS[:3]])  # 3 of 5, no repair allowed
    result = run_workflow("Prepare", provider=provider(Replies(partial), max_continuations=0), output_dir=tmp_path)
    assert result.metrics["appears_complete"] is False
    assert "Only 3 of 5 planned sections." in result.metrics["incomplete_reasons"]


# 3. output-token limit -----------------------------------------------------------------------------------
def test_output_limit_stop_is_identified_and_recorded(tmp_path):
    client = Replies(reply_text(), finish="length")
    with pytest.raises(ProviderError, match="output limit"):
        run_workflow("Prepare", provider=provider(client), output_dir=tmp_path, metrics_dir=tmp_path / "runs",
                     diagnostics_dir=tmp_path / "diag")
    assert len(client.calls) == 1  # a cut-off reply is never "continued" or retried
    failed = json.loads(next((tmp_path / "runs").glob("run-*.json")).read_text(encoding="utf-8"))
    assert failed["finish_reasons"] == ["length"]
    events = json.loads(next((tmp_path / "diag").iterdir()).read_text(encoding="utf-8"))["events"]
    assert events[0]["event"] == "output_limit_reached" and events[0]["finish_reason"] == "length"


# 4. parsing must not lose content --------------------------------------------------------------------------
REPEATED_KEYS = (
    '{"title":"T","meta_description":"M","sections":[' + ",".join(json.dumps(section(h)) for h in HEADINGS[:4]) + "],"
    '"faq":[{"question":"Q","answer":"A"}],"sections":[' + json.dumps(section("Conclusion")) + "]}"
)


def test_repeated_sections_key_is_merged_not_silently_dropped(tmp_path):
    """This is the reproduction of the observed symptom: plain json.loads keeps only 'Conclusion'."""
    assert [s["heading"] for s in json.loads(REPEATED_KEYS)["sections"]] == ["Conclusion"]
    client = Replies(REPEATED_KEYS)
    result = run_workflow("Prepare", provider=provider(client), output_dir=tmp_path, metrics_dir=tmp_path / "runs",
                          diagnostics_dir=tmp_path / "diag")
    assert [h for h, _ in result.article.sections] == HEADINGS  # all five survived
    assert len(client.calls) == 1  # no paid repair needed
    events = json.loads(open(result.metrics["diagnostics_file"], encoding="utf-8").read())["events"]
    assert events[0]["event"] == "repeated_json_keys" and events[0]["duplicate_keys"] == ["sections"]
    assert "Conclusion" in events[0]["raw_reply"]  # the raw reply is kept for diagnosis


def test_complete_reply_is_not_truncated_by_parsing_or_saving(tmp_path):
    long_body = "Sentence number one. " * 400  # 1,600 words in one section
    result = run_workflow("Prepare", provider=provider(Replies(reply_text([section(h, long_body) for h in HEADINGS]))),
                          output_dir=tmp_path)
    saved = open(result.output_path, encoding="utf-8").read()
    assert saved.count("Sentence number one.") == 400 * 5
    assert count_words(result.article) == run_seo_checks(result.article).word_count  # one consistent word count


# 5. bounded continuation ----------------------------------------------------------------------------------------
def test_continuation_fills_missing_sections_once_without_duplicates(tmp_path):
    first = reply_text([section("Introduction", "FIRST intro text. " * 70), section("Conclusion", "FIRST end. " * 70)])
    second = json.dumps({"sections": [section("Key Concepts to Know", "NEW concepts. " * 70),
                                      section("A Practical Study Plan", "NEW plan. " * 70),
                                      section("Common Mistakes to Avoid", "NEW mistakes. " * 70),
                                      section("Introduction", "DUPLICATE intro must be ignored. " * 30)]})
    client = Replies(first, second)
    result = run_workflow("Prepare", provider=provider(client), output_dir=tmp_path, metrics_dir=tmp_path / "runs",
                          diagnostics_dir=tmp_path / "diag")
    assert len(client.calls) == 2  # one article request + exactly one bounded follow-up
    assert [h for h, _ in result.article.sections] == HEADINGS  # outline order, five sections, no duplicates
    bodies = dict(result.article.sections)
    assert "FIRST intro" in bodies["Introduction"] and "DUPLICATE" not in " ".join(bodies.values())
    assert "NEW plan" in bodies["A Practical Study Plan"]
    follow_up = client.calls[1]["messages"][1]["content"]
    assert "Key Concepts to Know" in follow_up and "do not repeat" in follow_up.lower()
    m = result.metrics
    assert m["model_calls"] == 2 and m["continuation_calls"] == 1 and m["usage_source"] == "actual"
    assert m["input_tokens"] == 200 and m["output_tokens"] == 800  # both calls are counted
    assert m["appears_complete"] is True


def test_continuation_is_bounded_even_if_the_follow_up_also_falls_short(tmp_path):
    first = reply_text([section("Conclusion")])
    client = Replies(first, json.dumps({"sections": [section("Introduction")]}), reply_text([section("Conclusion")]))
    with pytest.raises(ProviderError, match="follow-up request"):
        run_workflow("Prepare", provider=provider(client), output_dir=tmp_path, metrics_dir=tmp_path / "runs")
    assert len(client.calls) == 2  # never a third request
    failed = json.loads(next((tmp_path / "runs").glob("run-*.json")).read_text(encoding="utf-8"))
    assert failed["model_calls"] == 2 and failed["status"] == "failed"


def test_continuation_can_be_turned_off_and_is_validated():
    assert get_max_continuations({}) == 1 and get_max_continuations({"MAX_CONTINUATION_CALLS": "0"}) == 0
    with pytest.raises(ConfigError):
        get_max_continuations({"MAX_CONTINUATION_CALLS": "9"})
    client = Replies(reply_text([section("Introduction"), section("Conclusion")]))
    p = provider(client, max_continuations=0)
    assert len(p.generate_sections("t", "k", create_outline("t", "k")[:2])) == 2
    assert len(client.calls) == 1


def test_unparseable_follow_up_keeps_what_exists_and_flags_it(tmp_path):
    first = reply_text([section(h) for h in HEADINGS[:3]])
    client = Replies(first, "this is not json")
    result = run_workflow("Prepare", provider=provider(client), output_dir=tmp_path)
    assert len(client.calls) == 2 and len(result.article.sections) == 3
    assert result.metrics["appears_complete"] is False


def test_merge_and_missing_helpers():
    outline = create_outline("t", "k")
    existing = [("Introduction", "a"), ("Conclusion", "z")]
    assert [o.heading for o in missing_headings(outline, existing)] == HEADINGS[1:4]
    merged = merge_sections(outline, existing, [("Key Concepts to Know", "b"), ("Introduction", "dup")])
    assert [h for h, _ in merged] == ["Introduction", "Key Concepts to Know", "Conclusion"]
    assert dict(merged)["Introduction"] == "a"
    renamed = [("Intro", "a"), ("Ideas", "b")]  # model renamed everything: assume it covers the first N
    assert [o.heading for o in missing_headings(outline, renamed)] == HEADINGS[2:]


# 6. incomplete output is never "complete"; exports stay blocked ----------------------------------------------------------
def test_word_count_is_consistent_between_completeness_and_seo_report(tmp_path):
    result = run_workflow("Prepare", provider=provider(Replies(reply_text())), output_dir=tmp_path)
    assert count_words(result.article) == result.seo_report.word_count == result.metrics["word_count"]
    short = reply_text([section(h, "Tiny body.") for h in HEADINGS])
    shortfall = run_workflow("Prepare", provider=provider(Replies(short), max_continuations=0), output_dir=tmp_path)
    complete, reasons = check_completeness(shortfall.article, create_outline("Prepare", "Prepare"), 900)
    assert complete is False and f"Only about {count_words(shortfall.article)} words for a 900-word target." in reasons


def test_incomplete_draft_keeps_markdown_and_pdf_blocked_and_the_same_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path / "o")
    monkeypatch.setattr(web, "METRICS_DIR", tmp_path / "r")
    short = reply_text([section(h, "Tiny body.") for h in HEADINGS])
    monkeypatch.setattr(web, "build_provider", lambda **kw: provider(Replies(short), max_continuations=0))
    c = TestClient(web.app, follow_redirects=False)
    slug = c.post("/generate", data={"topic": "Prepare", "keyword": "", "length": "medium"}) \
        .headers["location"].rsplit("/", 1)[1]
    assert c.get(f"/articles/{slug}/download").status_code == 409
    assert c.get(f"/articles/{slug}/download.pdf").status_code == 409
    assert "Export is blocked" in c.get(f"/articles/{slug}").text
    record = store.load_record(web.OUTPUT_DIR, slug)
    assert record["metrics"]["appears_complete"] is False
    assert c.post(f"/articles/{slug}/approve", data={"level": "export"}).status_code == 409


def test_web_failure_for_missing_sections_saves_diagnostics_and_shows_the_cause(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path / "o")
    monkeypatch.setattr(web, "METRICS_DIR", tmp_path / "r")
    one = reply_text([section("Conclusion")])
    monkeypatch.setattr(web, "build_provider", lambda **kw: provider(Replies(one), max_continuations=0))
    page = TestClient(web.app, follow_redirects=False).post(
        "/generate", data={"topic": "Prepare", "keyword": "", "length": "medium"})
    assert page.status_code == 502 and "only 1 of 5 requested sections" in page.text
    assert "not a token-limit problem" in page.text and KEY not in page.text
    diag = list((tmp_path / "diagnostics").glob("diag-*.json"))
    assert len(diag) == 1
    saved = json.loads(diag[0].read_text(encoding="utf-8"))
    assert saved["events"][0]["finish_reason"] == "stop" and saved["events"][0]["planned_sections"] == 5
    assert KEY not in diag[0].read_text(encoding="utf-8")  # no secrets, no prompts
