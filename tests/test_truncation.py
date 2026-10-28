"""Output-limit and finish-reason handling. Mocked responses only; no network, no credits."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.web as web
from app.config import build_provider, get_max_output_tokens, get_reasoning_effort
from app.models import ConfigError, ProviderError
from app.providers.groq_provider import GroqProvider
from app.providers.prompts import user_prompt
from app.workflow import create_outline, run_workflow

KEY = "gsk-trunc-test-not-real"


def article_json():
    return json.dumps({
        "title": "Step 1 Guide", "meta_description": "A usmle step 1 guide.",
        "sections": [{"heading": "A", "body": "One."}, {"heading": "B", "body": "Two."},
                     {"heading": "C", "body": "Three."},
                     {"heading": "D", "body": "Four."}, {"heading": "E", "body": "Five."}],
        "faq": [{"question": "Q?", "answer": "A."}],
    })


def response(text=None, finish="stop", completion=100, reasoning=None, omit_finish=False):
    choice = SimpleNamespace(message=SimpleNamespace(content=text or article_json()))
    if not omit_finish:
        choice.finish_reason = finish
    usage = SimpleNamespace(
        prompt_tokens=50, completion_tokens=completion, prompt_tokens_details=None,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def stub(resp):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return resp

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           calls=calls)


# Finish-reason semantics -----------------------------------------------------------------
def test_normal_completion_is_accepted(tmp_path):
    client = stub(response(finish="stop"))
    result = run_workflow("Step 1", provider=GroqProvider(KEY, "m", client=client), output_dir=tmp_path)
    assert result.metrics["appears_complete"] is True
    assert len(client.calls) == 1


def test_length_finish_reason_is_truncation_and_explains_how_to_fix(tmp_path):
    client = stub(response(finish="length", completion=3000, reasoning=2500))
    provider = GroqProvider(KEY, "m", max_tokens=3000, client=client)
    with pytest.raises(ProviderError) as info:
        run_workflow("Step 1", provider=provider, output_dir=tmp_path, metrics_dir=tmp_path / "runs")
    message = str(info.value)
    assert "3000 tokens" in message and "2500 of them reasoning" in message
    assert "MAX_OUTPUT_TOKENS" in message and "nothing was retried" in message
    assert len(client.calls) == 1  # no automatic second paid request
    assert not list(tmp_path.glob("*.md"))  # incomplete output is never saved as an article
    failed = json.loads(next((tmp_path / "runs").iterdir()).read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["output_tokens"] == 3000 and failed["reasoning_tokens"] == 2500
    assert failed["usage_source"] == "actual"


@pytest.mark.parametrize("kwargs", [{"omit_finish": True}, {"finish": None}, {"finish": "something_new"}])
def test_missing_or_unexpected_finish_reason_is_not_called_truncation(kwargs, tmp_path):
    client = stub(response(**kwargs))
    result = run_workflow("Step 1", provider=GroqProvider(KEY, "m", client=client), output_dir=tmp_path)
    assert result.metrics["status"] == "ok"


def test_broken_json_without_length_is_a_format_error_not_truncation():
    provider = GroqProvider(KEY, "m", client=stub(response(text='{"title": "cut', finish="stop")))
    with pytest.raises(ProviderError, match="expected JSON"):
        provider.generate_sections("t", "k", [])


def test_a_sentence_ending_is_not_used_to_guess_truncation(tmp_path):
    text = json.loads(article_json())
    text["sections"][-1]["body"] = "Ends without a full stop"
    client = stub(response(text=json.dumps(text), finish="stop"))
    result = run_workflow("Step 1", provider=GroqProvider(KEY, "m", client=client), output_dir=tmp_path)
    assert result.metrics["status"] == "ok"  # saved; the heuristic only adds a warning flag
    assert result.metrics["appears_complete"] is False


# Configurable limits ------------------------------------------------------------------------
def test_output_limit_comes_from_one_place():
    assert get_max_output_tokens({}) == 8000
    assert get_max_output_tokens({"MAX_OUTPUT_TOKENS": "12000"}) == 12000
    env = {"AI_PROVIDER": "groq", "GROQ_API_KEY": KEY, "GROQ_MODEL": "m", "MAX_OUTPUT_TOKENS": "9000"}
    assert build_provider(env).max_tokens == 9000
    assert build_provider({**env, "MAX_OUTPUT_TOKENS": ""}).max_tokens == 8000
    with pytest.raises(ConfigError):
        get_max_output_tokens({"MAX_OUTPUT_TOKENS": "-5"})


def test_limit_is_sent_to_the_api():
    client = stub(response())
    GroqProvider(KEY, "m", max_tokens=7777, client=client).generate_sections("t", "k", [])
    assert client.calls[0]["max_completion_tokens"] == 7777


def test_reasoning_effort_is_optional_and_validated():
    assert get_reasoning_effort({}) is None
    assert get_reasoning_effort({"GROQ_REASONING_EFFORT": "Low"}) == "low"
    with pytest.raises(ConfigError):
        get_reasoning_effort({"GROQ_REASONING_EFFORT": "extreme"})
    unset, low = stub(response()), stub(response())
    GroqProvider(KEY, "m", client=unset).generate_sections("t", "k", [])
    GroqProvider(KEY, "m", client=low, reasoning_effort="low").generate_sections("t", "k", [])
    assert "reasoning_effort" not in unset.calls[0]
    assert low.calls[0]["reasoning_effort"] == "low"


# Article length ------------------------------------------------------------------------------
def test_medium_length_instruction_targets_about_900_words():
    outline = create_outline("t", "k")
    text = user_prompt("optimized", "t", "k", outline, 900)
    assert "about 900 words" in text and "aim for 810-990" in text
    assert "words" not in user_prompt("optimized", "t", "k", outline)  # no length unless asked


def test_web_medium_selection_reaches_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path / "o")
    monkeypatch.setattr(web, "METRICS_DIR", tmp_path / "r")
    client = stub(response())
    monkeypatch.setattr(web, "build_provider", lambda **kw: GroqProvider(KEY, "m", client=client, **kw))
    page = TestClient(web.app, follow_redirects=False).post(
        "/generate", data={"topic": "Step 1", "keyword": "usmle step 1", "length": "medium"})
    assert page.status_code == 303
    assert "about 900 words" in client.calls[0]["messages"][1]["content"]


# UI --------------------------------------------------------------------------------------------
def test_ui_shows_truncation_error_and_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path / "o")
    monkeypatch.setattr(web, "METRICS_DIR", tmp_path / "r")
    client = stub(response(finish="length", completion=4000, reasoning=3500))
    monkeypatch.setattr(web, "build_provider",
                        lambda **kw: GroqProvider(KEY, "m", client=client, max_tokens=4000, **kw))
    page = TestClient(web.app, follow_redirects=False).post(
        "/generate", data={"topic": "Step 1", "keyword": "usmle step 1", "length": "medium"})
    assert page.status_code == 502
    assert "cut off" in page.text and "MAX_OUTPUT_TOKENS" in page.text
    assert KEY not in page.text
    assert not list((tmp_path / "o").glob("*.md")) if (tmp_path / "o").exists() else True
    assert len(client.calls) == 1
