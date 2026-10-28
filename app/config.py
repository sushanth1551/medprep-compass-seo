"""Reads settings from environment variables (and a local .env file).

Secrets are only ever read here; nothing in this module prints them.
"""

import os

from app.models import DEFAULT_MAX_OUTPUT_TOKENS, ConfigError
from app.providers.prompts import DEFAULT_PROMPT_STYLE, PROMPT_STYLES

SUPPORTED_PROVIDERS = ("mock", "claude", "groq")
REASONING_EFFORTS = ("low", "medium", "high")

# provider -> (key variable, model variable, timeout variable)
_REAL_PROVIDER_VARS = {
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_TIMEOUT_SECONDS"),
    "groq": ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_TIMEOUT_SECONDS"),
}


def load_local_env():
    """Load a local .env file if python-dotenv is installed. Real env vars win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def get_provider_name(env=None):
    env = os.environ if env is None else env
    name = (env.get("AI_PROVIDER") or "mock").strip().lower()
    if name not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"AI_PROVIDER='{name}' is not supported. Use one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return name


def get_max_output_tokens(env=None):
    env = os.environ if env is None else env
    raw = (env.get("MAX_OUTPUT_TOKENS") or "").strip()
    if not raw:
        return DEFAULT_MAX_OUTPUT_TOKENS
    if not raw.isdigit() or int(raw) <= 0:
        raise ConfigError("MAX_OUTPUT_TOKENS must be a positive whole number.")
    return int(raw)


def get_reasoning_effort(env=None):
    """Optional GROQ_REASONING_EFFORT. None = do not send the parameter at all."""
    env = os.environ if env is None else env
    value = (env.get("GROQ_REASONING_EFFORT") or "").strip().lower()
    if value and value not in REASONING_EFFORTS:
        raise ConfigError(f"GROQ_REASONING_EFFORT must be one of: {', '.join(REASONING_EFFORTS)}.")
    return value or None


def get_max_continuations(env=None):
    """Extra requests allowed to write sections the first reply left out (0 turns it off)."""
    env = os.environ if env is None else env
    raw = (env.get("MAX_CONTINUATION_CALLS") or "").strip()
    if not raw:
        return 1
    if not raw.isdigit() or int(raw) > 2:
        raise ConfigError("MAX_CONTINUATION_CALLS must be 0, 1 or 2.")
    return int(raw)


def get_prompt_style(env=None):
    env = os.environ if env is None else env
    style = (env.get("PROMPT_STYLE") or DEFAULT_PROMPT_STYLE).strip().lower()
    if style not in PROMPT_STYLES:
        raise ConfigError(f"PROMPT_STYLE must be one of: {', '.join(PROMPT_STYLES)}.")
    return style


def regenerate_enabled(env=None):
    """Targeted rewrite of empty sections is OFF unless explicitly turned on."""
    env = os.environ if env is None else env
    return (env.get("REGENERATE_FAILED_SECTIONS") or "").strip().lower() in ("1", "true", "yes")


def build_provider(env=None, prompt_style=None, target_words=None):
    """Return the provider chosen by AI_PROVIDER. Raises ConfigError if misconfigured."""
    env = os.environ if env is None else env
    name = get_provider_name(env)
    style = prompt_style or get_prompt_style(env)
    if name == "mock":
        from app.providers.mock_provider import MockProvider

        return MockProvider(prompt_style=style, target_words=target_words)

    key_var, model_var, timeout_var = _REAL_PROVIDER_VARS[name]
    api_key = (env.get(key_var) or "").strip()
    model = (env.get(model_var) or "").strip()
    missing = [n for n, v in ((key_var, api_key), (model_var, model)) if not v]
    if missing:
        raise ConfigError(
            f"AI_PROVIDER={name} needs {' and '.join(missing)}. Copy .env.example to .env, "
            "fill the values in yourself (never paste a key into chat), or set "
            "AI_PROVIDER=mock to work offline."
        )
    try:
        timeout = float(env.get(timeout_var) or 120)
    except ValueError:
        raise ConfigError(f"{timeout_var} must be a number.") from None
    max_tokens = get_max_output_tokens(env)

    if name == "claude":
        from app.providers.claude_provider import ClaudeProvider as Provider
    else:
        from app.providers.groq_provider import GroqProvider as Provider
    extra = {"reasoning_effort": get_reasoning_effort(env)} if name == "groq" else {}
    extra["max_continuations"] = get_max_continuations(env)
    return Provider(api_key=api_key, model=model, timeout=timeout, max_tokens=max_tokens,
                    prompt_style=style, target_words=target_words, **extra)


def output_limit_warning(max_tokens, target_words):
    """A plain-language warning if the output limit looks too small for the chosen length.

    Rough rule: English needs about 1.3-1.5 tokens per word and the JSON wrapper, headings and
    a reasoning model's thinking add more, so we ask for 3 tokens per target word.
    """
    needed = int(target_words * 3)
    if max_tokens < needed:
        return (
            f"MAX_OUTPUT_TOKENS is {max_tokens}, which may be too low for about {target_words} words "
            f"(try at least {needed}). Words and tokens are different measurements."
        )
    return None
