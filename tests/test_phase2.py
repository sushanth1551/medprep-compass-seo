"""Phase 2 tests. No test calls the network or uses a real key."""

import json
from types import SimpleNamespace

import anthropic
import pytest

try:
    import httpx2 as httpx  # the SDK's HTTP library in newer releases
except ImportError:  # pragma: no cover
    import httpx

from app.config import build_provider
from app.models import Article, ConfigError, ProviderError
from app.providers.claude_provider import ClaudeProvider
from app.providers.mock_provider import MockProvider
from app.seo_checks import run_seo_checks
from app.token_tracker import TokenTracker
from app.workflow import run_workflow

FAKE_KEY = "test-key-not-real"
REQUEST = httpx.Request("POST", "https://example.invalid/v1/messages")


def good_json():
    return json.dumps(
        {
            "title": "Step 1 Study Plan: A Practical Guide",
            "meta_description": "Build a realistic step 1 study plan with weekly goals.",
            "sections": [
                {"heading": "Introduction", "body": "A step 1 study plan helps you organise time."},
                {"heading": "Key Concepts to Know", "body": "Know the main ideas."},
                {"heading": "A Practical Study Plan", "body": "Split each week into review and practice."},
                {"heading": "Common Mistakes to Avoid", "body": "Avoid cramming."},
                {"heading": "Conclusion", "body": "Adjust the plan as you learn."},
            ],
            "faq": [{"question": "How long?", "answer": "It depends on your schedule."}],
        }
    )


def fake_client(text=None, error=None, stop_reason="end_turn"):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(
            content=[SimpleNamespace(text=text if text is not None else good_json())],
            usage=SimpleNamespace(input_tokens=111, output_tokens=222),
            stop_reason=stop_reason,
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create), calls=calls)


def status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


# Configuration ---------------------------------------------------------------
def test_mock_mode_needs_no_credentials():
    assert isinstance(build_provider({}), MockProvider)
    assert isinstance(build_provider({"AI_PROVIDER": "mock"}), MockProvider)


def test_claude_mode_without_key_is_rejected_safely():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_provider({"AI_PROVIDER": "claude"})


def test_claude_mode_without_model_is_rejected():
    with pytest.raises(ConfigError, match="ANTHROPIC_MODEL"):
        build_provider({"AI_PROVIDER": "claude", "ANTHROPIC_API_KEY": FAKE_KEY})


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError):
        build_provider({"AI_PROVIDER": "other"})


def test_error_message_never_contains_key():
    with pytest.raises(ConfigError) as info:
        build_provider({"AI_PROVIDER": "claude", "ANTHROPIC_API_KEY": FAKE_KEY})
    assert FAKE_KEY not in str(info.value)


# Claude provider (mocked client) -------------------------------------------------
def test_claude_provider_success_records_actual_usage(tmp_path):
    client = fake_client()
    provider = ClaudeProvider(FAKE_KEY, "some-model", client=client)
    result = run_workflow("Step 1 study plan", "step 1 study plan", provider, tmp_path)
    assert len(client.calls) == 1  # one request only
    assert result.usage.actual_input_tokens == 111
    assert result.usage.actual_output_tokens == 222
    assert result.usage.is_estimate is False
    assert "ACTUAL" in result.usage.source_label
    assert result.usage.usage_source == "actual"
    assert result.usage.estimated_cost is None  # no invented pricing
    assert "AI-GENERATED DRAFT" in result.article.to_markdown()


@pytest.mark.parametrize(
    "error, text",
    [
        (status_error(anthropic.AuthenticationError, 401), "authentication"),
        (status_error(anthropic.RateLimitError, 429), "rate limit"),
        (anthropic.APITimeoutError(request=REQUEST), "timed out"),
        (anthropic.APIConnectionError(request=REQUEST), "network"),
        (status_error(anthropic.InternalServerError, 500), "500"),
    ],
)
def test_provider_errors_are_clear_and_not_retried(error, text, tmp_path):
    client = fake_client(error=error)
    provider = ClaudeProvider(FAKE_KEY, "some-model", client=client)
    with pytest.raises(ProviderError, match=f"(?i){text}") as info:
        run_workflow("Anatomy", provider=provider, output_dir=tmp_path)
    assert len(client.calls) == 1
    assert FAKE_KEY not in str(info.value)
    assert list(tmp_path.iterdir()) == []  # no fallback draft was written


def test_bad_json_and_truncation_rejected():
    for client in (fake_client(text="not json"), fake_client(stop_reason="max_tokens")):
        provider = ClaudeProvider(FAKE_KEY, "m", client=client)
        with pytest.raises(ProviderError):
            provider.generate_sections("t", "k", [])


# Tokens ------------------------------------------------------------------------
def test_mock_usage_is_labelled_estimate(tmp_path):
    usage = run_workflow("Anatomy", output_dir=tmp_path).usage
    assert usage.is_estimate is True
    assert usage.actual_input_tokens is None
    assert "ESTIMATE" in usage.source_label


def test_cost_uses_actual_tokens_only_when_prices_given():
    tracker = TokenTracker(price_per_1k_input=1.0, price_per_1k_output=2.0)
    tracker.record_actual(1000, 500)
    usage = tracker.summary()
    assert usage.estimated_cost == pytest.approx(2.0)
    assert "API-reported" in usage.cost_note


# SEO checks ----------------------------------------------------------------------
def make_article(**overrides):
    base = dict(
        title="Step 1 Study Plan Guide",
        meta_description="A clear step 1 study plan guide.",
        keyword="step 1 study plan",
        sections=[("One", "Plan your week."), ("Two", "Review daily."), ("Three", "Practice often.")],
        faq=[("Q?", "A.")],
    )
    base.update(overrides)
    return Article(**base)


def test_valid_draft_has_no_failures():
    report = run_seo_checks(make_article())
    assert report.failed == []
    assert report.word_count > 0
    assert "not a Google ranking score" in report.score_text
    assert "NOT PERFORMED" in report.source_verification
    assert report.needs_human_review is True


def test_invalid_draft_fails_and_warns():
    report = run_seo_checks(
        make_article(title="", meta_description="", sections=[("One", "x")], faq=[])
    )
    text = " ".join(report.failed)
    assert "Title is empty" in text
    assert "Meta description is empty" in text
    assert "fewer than 3" in text
    assert "FAQ section is missing" in text


def test_missing_keyword_is_a_warning_not_a_failure():
    report = run_seo_checks(make_article(title="Something else", meta_description="Other text."))
    assert any("title" in w for w in report.warnings)
    assert any("meta description" in w for w in report.warnings)
    assert report.failed == []


def test_exam_promise_fails():
    sections = [("One", "We guarantee you pass."), ("Two", "x y."), ("Three", "z.")]
    report = run_seo_checks(make_article(sections=sections))
    assert any("promise" in f for f in report.failed)


def test_claims_and_references_flagged_but_never_verified():
    article = make_article(
        sections=[
            ("One", "Studies show 80% of students improve. See https://example.com/x."),
            ("Two", "Review daily."),
            ("Three", "Practice often."),
        ]
    )
    report = run_seo_checks(article)
    assert report.claims_needing_evidence
    assert report.references_found == ["https://example.com/x."] or report.references_found
    assert any("NOT been verified" in w for w in report.warnings)


def test_output_files_never_overwritten(tmp_path):
    first = run_workflow("Anatomy", output_dir=tmp_path)
    second = run_workflow("Anatomy", output_dir=tmp_path)
    assert first.output_path != second.output_path
    assert json.loads(open(first.report_path, encoding="utf-8").read())["needs_human_review"] is True
