"""Web interface tests. They use FastAPI's test client: no server, no network, no credits."""

import json
from types import SimpleNamespace

import groq
import pytest
from fastapi.testclient import TestClient

try:
    import httpx
except ImportError:  # pragma: no cover
    import httpx2 as httpx

import app.web as web
from app.config import build_provider
from app.providers.groq_provider import GroqProvider

FAKE_KEY = "gsk-web-test-key-not-real"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(web, "METRICS_DIR", tmp_path / "runs")
    for name in ("GROQ_API_KEY", "GROQ_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_PROVIDER", "mock")
    return TestClient(web.app, follow_redirects=False)


def article_json():
    return json.dumps({
        "title": "Study Schedule Guide", "meta_description": "A study schedule guide.",
        "sections": [{"heading": "Introduction", "body": "Plan your weeks."},
                     {"heading": "Key Concepts to Know", "body": "Know the ideas."},
                     {"heading": "A Practical Study Plan", "body": "Review daily."},
                     {"heading": "Common Mistakes to Avoid", "body": "Avoid cramming."},
                     {"heading": "Conclusion", "body": "Adjust as you go."}],
        "faq": [{"question": "Q?", "answer": "A."}],
    })


def stub_groq(monkeypatch, error=None):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=article_json()),
                                     finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=60, prompt_tokens_details=None),
        )

    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(
        web, "build_provider",
        lambda **kw: GroqProvider(FAKE_KEY, "my-model", client=stub, **kw),
    )
    monkeypatch.setenv("AI_PROVIDER", "groq")
    return calls


def post(client, **fields):
    data = {"topic": "USMLE Step 1 study plan", "keyword": "step 1 study plan", "length": "short"}
    data.update(fields)
    return client.post("/generate", data=data)


# Pages ---------------------------------------------------------------------------------
def test_index_shows_provider_and_form(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "mock" in page.text and 'name="topic"' in page.text and 'name="length"' in page.text
    assert "No drafts yet" in page.text


def test_mock_generation_end_to_end(client):
    response = post(client)
    assert response.status_code == 303
    slug = response.headers["location"].rsplit("/", 1)[1]
    page = client.get(f"/articles/{slug}")
    assert page.status_code == 200
    for needle in ("SEO checklist", "ESTIMATE only", "Meta description", "Frequently Asked",
                   "cannot guarantee", "unknown", "NOT PERFORMED"):
        assert needle in page.text
    download = client.get(f"/articles/{slug}/download")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]
    assert download.text.startswith("# ")
    assert list((web.OUTPUT_DIR).glob("*.web.json"))
    assert list(web.METRICS_DIR.glob("run-*.json"))
    assert slug in client.get("/").text  # listed under previous drafts


def test_drafts_are_never_overwritten(client):
    first = post(client).headers["location"]
    second = post(client).headers["location"]
    assert first != second
    assert len(list(web.OUTPUT_DIR.glob("*.md"))) == 2


def test_legacy_markdown_draft_is_viewable(client):
    web.OUTPUT_DIR.mkdir()
    (web.OUTPUT_DIR / "old-draft.md").write_text("# Old draft\n\ntext", encoding="utf-8")
    page = client.get("/articles/old-draft")
    assert page.status_code == 200 and "Old draft" in page.text


# Validation and routing -------------------------------------------------------------------
def test_empty_topic_rejected_without_creating_files(client):
    response = post(client, topic="   ")
    assert response.status_code == 400 and "Topic cannot be empty" in response.text
    assert not web.OUTPUT_DIR.exists() or not list(web.OUTPUT_DIR.iterdir())


def test_overlong_keyword_and_bad_length_rejected(client):
    assert post(client, keyword="k" * 500).status_code == 400
    assert post(client, length="enormous").status_code == 400


@pytest.mark.parametrize("path", ["/articles/does-not-exist", "/articles/BAD_NAME",
                                  "/articles/..%2Fsecret", "/articles/x/download"])
def test_unknown_or_unsafe_article_names_are_404(client, path):
    assert client.get(path).status_code == 404


def test_second_request_while_busy_is_refused(client):
    assert web._generation_lock.acquire(blocking=False)
    try:
        response = post(client)
    finally:
        web._generation_lock.release()
    assert response.status_code == 429 and "already running" in response.text


# Providers, secrets and errors -----------------------------------------------------------------
def test_groq_without_credentials_shows_setup_help_and_exits_safely(client, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    assert "GROQ_API_KEY" in client.get("/").text
    response = post(client)
    assert response.status_code == 500 and "GROQ_API_KEY" in response.text


def test_groq_success_shows_actual_usage_and_never_leaks_key(client, monkeypatch):
    calls = stub_groq(monkeypatch)
    response = post(client, length="long")
    assert response.status_code == 303
    assert len(calls) == 1  # exactly one paid request
    assert "about 1500 words" in calls[0]["messages"][1]["content"]
    page = client.get(response.headers["location"])
    assert "ACTUAL (from the API): input 120, output 60" in page.text
    assert "my-model" in page.text and "groq" in page.text
    assert "unknown" in page.text  # cost unknown without pricing
    everything = page.text + client.get("/").text
    everything += "".join(p.read_text(encoding="utf-8") for p in web.OUTPUT_DIR.iterdir())
    assert FAKE_KEY not in everything


def test_api_failure_shows_clear_error_without_fallback_or_retry(client, monkeypatch):
    error = groq.RateLimitError(
        "boom", response=httpx.Response(429, request=httpx.Request("POST", "https://x.invalid")),
        body=None,
    )
    calls = stub_groq(monkeypatch, error=error)
    response = post(client)
    assert response.status_code == 502
    assert "rate limit" in response.text.lower() and "nothing was retried" in response.text.lower()
    assert FAKE_KEY not in response.text
    assert len(calls) == 1
    assert not list(web.OUTPUT_DIR.glob("*.md"))  # no mock draft substituted
    failed = json.loads(next(web.METRICS_DIR.iterdir()).read_text(encoding="utf-8"))
    assert failed["status"] == "failed"


def test_index_uses_real_config_function_in_mock_mode(client):
    assert build_provider().provider_id == "mock"
