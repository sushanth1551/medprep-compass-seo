"""Cost calculation, kept apart from the workflow.

Rates live in pricing.json (USD per 1,000,000 tokens, per "provider/model-id").
This project ships NO rates: you add them yourself from the provider's pricing page.
Without a rate for the exact model, cost is reported as unknown (None), never zero.
"""

import json
from pathlib import Path

DEFAULT_PRICING_FILE = Path(__file__).resolve().parent.parent / "pricing.json"


def load_pricing(path=DEFAULT_PRICING_FILE):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"models": {}}


def _unknown(note, currency="USD"):
    return {"amount": None, "currency": currency, "status": "unknown", "note": note}


def compute_cost(provider_id, model, usage_source, input_tokens, output_tokens,
                 cached_tokens=None, pricing=None):
    """Return {"amount", "currency", "status", "note"}. amount is None when unknown."""
    pricing = load_pricing() if pricing is None else pricing
    currency = pricing.get("currency", "USD")
    if usage_source != "actual":
        return _unknown(f"Cost not calculated: token usage is {usage_source}, not actual.", currency)
    rates = (pricing.get("models") or {}).get(f"{provider_id}/{model}")
    if not rates or "input" not in rates or "output" not in rates:
        return _unknown(f"No rates for '{provider_id}/{model}' in pricing.json.", currency)
    cached = cached_tokens or 0
    if cached and "cached_input" not in rates:
        return _unknown("Cached tokens were used but pricing.json has no cached_input rate.", currency)
    amount = (
        (input_tokens - cached) * rates["input"]
        + cached * rates.get("cached_input", 0)
        + output_tokens * rates["output"]
    ) / 1_000_000
    checked = pricing.get("checked_on") or "date not recorded"
    return {
        "amount": amount,
        "currency": currency,
        "status": "known",
        "note": f"Computed from pricing.json rates (per 1M tokens), checked on {checked}. Not a bill.",
    }
