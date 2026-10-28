"""Phase 3 tests: Groq, metrics, prompts, truncation, cost, benchmark. No network."""

import json
from types import SimpleNamespace

import groq
import pytest

try:
    import httpx
except ImportError:  # pragma: no cover
    import httpx2 as httpx

from app.benchmark import compare, render_markdown, run_benchmark
from app.config import build_provider, get_max_output_tokens
from app.metrics import check_completeness
from app.models import DEFAULT_MAX_OUTPUT_TOKENS, Article, ConfigError, ProviderError
from app.pricing import compute_cost
from app.providers.groq_provider import GroqProvider
from app.providers.mock_provider import MockProvider
from app.providers.prompts import BASELINE_SYSTEM, OPTIMIZED_SYSTEM
from app.token_tracker import TokenTracker
from app.workflow import create_outline, run_workflow

FAKE_KEY = "gsk-test-key-not-real"
REQUEST = httpx.Request("POST", "https://example.invalid/chat")


def article_json(empty_body=False):
    return json.dumps(
        {
            "title": "Study Schedule Guide",
            "meta_description": "A study schedule guide.",
            "sections": [
                {"heading": "Introduction", "body": "Plan your weeks."},
                {"heading": "Key Concepts to Know", "body": "Know the ideas."},
                {"heading": "A Practical Study Plan", "body": "" if empty_body else "Review daily."},
                {"heading": "Common Mistakes to Avoid", "body": "Avoid cramming."},
                {"heading": "Conclusion", "body": "Adjust as you go."},
            ],
            "faq": [{"question": "Q?", "answer": "A."}],
        }
    )


def groq_response(text=None, finish="stop", usage="default"):
    if usage == "default":
        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text or article_json()),
                                 finish_reason=finish)],
        usage=usage,
    )


def groq_client(*responses, error=None):
    calls, queue = [], list(responses)

    def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return queue.pop(0)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           calls=calls)


def status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


# Configuration -----------------------------------------------------------------
def test_groq_selected_with_env_values():
    provider = build_provider({"AI_PROVIDER": "groq", "GROQ_API_KEY": FAKE_KEY,
                               "GROQ_MODEL": "my-model"})
    assert isinstance(provider, GroqProvider)
    assert provider.model == "my-model"  # exactly what was configured
    assert FAKE_KEY not in provider.name


@pytest.mark.parametrize("env, missing", [
    ({"AI_PROVIDER": "groq"}, "GROQ_API_KEY"),
    ({"AI_PROVIDER": "groq", "GROQ_API_KEY": FAKE_KEY}, "GROQ_MODEL"),
])
def test_groq_missing_configuration(env, missing):
    with pytest.raises(ConfigError, match=missing) as info:
        build_provider(env)
    assert FAKE_KEY not in str(info.value)


def test_claude_and_mock_still_selectable():
    assert isinstance(build_provider({}), MockProvider)
    assert build_provider({"AI_PROVIDER": "claude", "ANTHROPIC_API_KEY": FAKE_KEY,
                           "ANTHROPIC_MODEL": "m"}).provider_id == "claude"


def test_output_limit_is_configurable():
    assert get_max_output_tokens({}) == DEFAULT_MAX_OUTPUT_TOKENS == 8000
    assert get_max_output_tokens({"MAX_OUTPUT_TOKENS": "1500"}) == 1500
    with pytest.raises(ConfigError):
        get_max_output_tokens({"MAX_OUTPUT_TOKENS": "abc"})
    client = groq_client(groq_response())
    GroqProvider(FAKE_KEY, "m", max_tokens=1234, client=client).generate_sections("t", "k", [])
    assert client.calls[0]["max_completion_tokens"] == 1234
    assert client.calls[0]["model"] == "m"


# Groq provider ---------------------------------------------------------------------
def test_groq_extracts_actual_usage_and_counts_calls(tmp_path):
    client = groq_client(groq_response())
    provider = GroqProvider(FAKE_KEY, "my-model", client=client)
    result = run_workflow("Study schedule", provider=provider, output_dir=tmp_path,
                          metrics_dir=tmp_path / "runs")
    assert len(client.calls) == 1
    assert result.usage.usage_source == "actual"
    assert (result.usage.actual_input_tokens, result.usage.actual_output_tokens) == (100, 50)
    assert result.usage.cached_input_tokens == 20
    m = result.metrics
    assert (m["provider"], m["model"], m["model_calls"]) == ("groq", "my-model", 1)
    assert m["usage_source"] == "actual"
    assert FAKE_KEY not in json.dumps(m)


def test_missing_usage_is_unavailable_not_invented(tmp_path):
    provider = GroqProvider(FAKE_KEY, "m", client=groq_client(groq_response(usage=None)))
    result = run_workflow("Study schedule", provider=provider, output_dir=tmp_path)
    assert result.usage.usage_source == "unavailable"
    assert result.metrics["input_tokens"] is None
    assert result.metrics["cost"]["amount"] is None


@pytest.mark.parametrize("error, text", [
    (status_error(groq.AuthenticationError, 401), "authentication"),
    (status_error(groq.RateLimitError, 429), "rate limit"),
    (status_error(groq.NotFoundError, 404), "GROQ_MODEL"),
    (status_error(groq.BadRequestError, 400), "GROQ_MODEL"),
    (groq.APITimeoutError(request=REQUEST), "timed out"),
    (groq.APIConnectionError(request=REQUEST), "network"),
    (status_error(groq.InternalServerError, 500), "500"),
])
def test_groq_errors_are_clear_not_retried_and_leak_nothing(error, text, tmp_path):
    client = groq_client(error=error)
    provider = GroqProvider(FAKE_KEY, "m", client=client)
    with pytest.raises(ProviderError, match=f"(?i){text}") as info:
        run_workflow("Anatomy", provider=provider, output_dir=tmp_path)
    assert len(client.calls) == 1
    assert FAKE_KEY not in str(info.value)
    assert list(tmp_path.iterdir()) == []  # no mock fallback draft


def test_failed_run_still_records_metrics(tmp_path):
    provider = GroqProvider(FAKE_KEY, "m", client=groq_client(error=status_error(groq.RateLimitError, 429)))
    with pytest.raises(ProviderError):
        run_workflow("Anatomy", provider=provider, output_dir=tmp_path / "d", metrics_dir=tmp_path / "r")
    saved = json.loads(next((tmp_path / "r").iterdir()).read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and saved["model_calls"] == 1


# Truncation / completeness ---------------------------------------------------------------
def test_length_finish_reason_is_rejected_as_truncated(tmp_path):
    provider = GroqProvider(FAKE_KEY, "m", client=groq_client(groq_response(finish="length")))
    with pytest.raises(ProviderError, match="cut off"):
        run_workflow("Anatomy", provider=provider, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def make_article(sections, faq=(("Q?", "A."),)):
    return Article(title="T", meta_description="M", keyword="k", sections=sections, faq=list(faq))


def test_completeness_detection():
    outline = create_outline("topic", "k")
    ok = make_article([(o.heading, "Fine.") for o in outline])
    assert check_completeness(ok, outline) == (True, [])
    short = make_article([("A", "Fine.")])
    assert check_completeness(short, outline)[0] is False
    cut = make_article([(o.heading, "Fine.") for o in outline[:-1]] + [("End", "and then the")])
    complete, reasons = check_completeness(cut, outline[:-1] + outline[-1:])
    assert complete is False and any("cut-off" in r for r in reasons)


# Prompts --------------------------------------------------------------------------------
def test_optimized_prompt_is_shorter_but_keeps_essentials():
    assert len(OPTIMIZED_SYSTEM) < len(BASELINE_SYSTEM)
    for needle in ("citations", "JSON", "keyword", "exam"):
        assert needle in OPTIMIZED_SYSTEM
    outline = create_outline("topic", "kw")
    short = MockProvider("optimized").build_prompt_text("topic", "kw", outline)
    long = MockProvider("baseline").build_prompt_text("topic", "kw", outline)
    assert len(short) < len(long)


# Targeted regeneration ---------------------------------------------------------------------
def test_no_second_call_by_default_for_failed_section(tmp_path):
    client = groq_client(groq_response(article_json(empty_body=True)))
    result = run_workflow("Study", provider=GroqProvider(FAKE_KEY, "m", client=client),
                          output_dir=tmp_path)
    assert len(client.calls) == 1
    assert result.metrics["regenerated_sections"] == []
    assert any("no body" in f for f in result.seo_report.failed)


def test_regeneration_only_rewrites_the_empty_section_when_enabled(tmp_path):
    client = groq_client(
        groq_response(article_json(empty_body=True)),
        groq_response('{"body": "Rewritten weekly section."}'),
    )
    result = run_workflow("Study", provider=GroqProvider(FAKE_KEY, "m", client=client),
                          output_dir=tmp_path, regenerate_failed=True)
    assert len(client.calls) == 2  # the article plus ONE section, never a full rewrite
    assert result.metrics["model_calls"] == 2
    assert result.metrics["regenerated_sections"] == ["A Practical Study Plan"]
    assert dict(result.article.sections)["A Practical Study Plan"] == "Rewritten weekly section."
    assert result.usage.actual_input_tokens == 200  # both calls summed


# Metrics ---------------------------------------------------------------------------------------
def test_metrics_never_overwrite_earlier_runs(tmp_path):
    for _ in range(3):
        run_workflow("Anatomy", output_dir=tmp_path / "d", metrics_dir=tmp_path / "runs")
    files = list((tmp_path / "runs").iterdir())
    assert len(files) == 3
    data = json.loads(files[0].read_text(encoding="utf-8"))
    for key in ("topic", "provider", "model", "usage_source", "input_tokens", "output_tokens",
                "model_calls", "elapsed_seconds", "word_count", "seo", "appears_complete",
                "human_review_flags", "cost"):
        assert key in data


def test_mock_metrics_are_labelled_estimated_with_zero_model_calls(tmp_path):
    m = run_workflow("Anatomy", output_dir=tmp_path).metrics
    assert m["usage_source"] == "estimated"
    assert m["model_calls"] == 0
    assert m["cost"]["amount"] is None


def test_estimate_and_actual_labels_differ():
    est = TokenTracker()
    est.record_estimate("abc", "def")
    act = TokenTracker()
    act.record_actual(5, 6)
    assert est.summary().usage_source == "estimated"
    assert act.summary().usage_source == "actual"
    assert "ESTIMATE" in est.summary().source_label
    assert "ACTUAL" in act.summary().source_label


# Cost -----------------------------------------------------------------------------------------------
PRICING = {"currency": "USD", "checked_on": "2000-01-01",
           "models": {"groq/m": {"input": 2.0, "output": 4.0, "cached_input": 1.0}}}


def test_cost_with_known_rates():
    cost = compute_cost("groq", "m", "actual", 1_000_000, 500_000, pricing=PRICING)
    assert cost["status"] == "known"
    assert cost["amount"] == pytest.approx(4.0)  # 2.0 input + 2.0 output
    cached = compute_cost("groq", "m", "actual", 1_000_000, 0, cached_tokens=500_000, pricing=PRICING)
    assert cached["amount"] == pytest.approx(1.5)  # 0.5M*2.0 + 0.5M*1.0


def test_missing_pricing_is_unknown_not_zero():
    assert compute_cost("groq", "other", "actual", 10, 10, pricing=PRICING)["amount"] is None
    assert compute_cost("groq", "m", "actual", 10, 10, pricing={"models": {}})["status"] == "unknown"
    assert compute_cost("groq", "m", "estimated", 10, 10, pricing=PRICING)["amount"] is None
    assert compute_cost("groq", "m", "unavailable", 0, 0, pricing=PRICING)["amount"] is None


def test_workflow_cost_uses_pricing_for_exact_model(tmp_path):
    provider = GroqProvider(FAKE_KEY, "m", client=groq_client(groq_response()))
    result = run_workflow("Study", provider=provider, output_dir=tmp_path, pricing=PRICING)
    assert result.metrics["cost"]["status"] == "known"


# Benchmark ------------------------------------------------------------------------------------------
def test_mock_benchmark_writes_json_and_markdown(tmp_path):
    report, json_path, md_path = run_benchmark("mock", out_dir=tmp_path)
    assert len(report["runs"]) == 6 and len(report["comparisons"]) == 3
    assert "MOCK" in report["label"]
    assert json.loads(json_path.read_text(encoding="utf-8"))["mode"] == "mock"
    text = md_path.read_text(encoding="utf-8")
    assert "not evidence of real savings" in text and "cannot guarantee" in text
    assert all(r["usage_source"] == "estimated" and r["model_calls"] == 0 for r in report["runs"])
    # a second benchmark never overwrites the first
    run_benchmark("mock", out_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_live_benchmark_refuses_mock_provider(tmp_path):
    with pytest.raises(ConfigError):
        run_benchmark("live", env={"AI_PROVIDER": "mock"}, out_dir=tmp_path)


def run_metrics(**over):
    base = {"topic": "t", "status": "ok", "provider": "groq", "model": "m", "usage_source": "actual",
            "input_tokens": 1000, "output_tokens": 800, "model_calls": 1, "elapsed_seconds": 2.0,
            "word_count": 500, "appears_complete": True, "seo": {"failed": 0},
            "prompt_style": "baseline"}
    base.update(over)
    return base


def test_compare_reports_change_only_when_valid():
    ok = compare(run_metrics(), run_metrics(input_tokens=700), mock=False)
    assert ok["valid"] and ok["percent_change"]["input_tokens"] == -30.0
    mixed = compare(run_metrics(), run_metrics(usage_source="estimated"), mock=False)
    assert not mixed["valid"] and mixed["percent_change"] is None


def test_compare_rejects_savings_from_shorter_or_incomplete_articles():
    shorter = compare(run_metrics(), run_metrics(output_tokens=100, word_count=100), mock=False)
    assert not shorter["valid"]
    incomplete = compare(run_metrics(), run_metrics(appears_complete=False), mock=False)
    assert not incomplete["valid"]
    worse_seo = compare(run_metrics(), run_metrics(seo={"failed": 2}), mock=False)
    assert not worse_seo["valid"]


def test_markdown_renderer_handles_aborted_report():
    text = render_markdown({"mode": "live", "label": "LIVE API RUN", "aborted": "Stopped: boom",
                           "runs": [], "comparisons": [], "caveats": ["x"]})
    assert "ABORTED" in text


# Review follow-ups ---------------------------------------------------------------------------------
def test_prompts_ask_for_keyword_in_title_and_meta():
    for text in (BASELINE_SYSTEM, OPTIMIZED_SYSTEM):
        assert "meta description" in text and "title" in text


def test_outline_headings_do_not_repeat_a_long_topic():
    long_topic = "How to Prepare for USMLE Step 1: A Complete Study Plan for Medical Students"
    headings = [o.heading for o in create_outline(long_topic, long_topic)]
    assert all(long_topic not in h for h in headings)
    assert len(set(headings)) == len(headings)
