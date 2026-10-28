"""Rough token estimation. Everything here is an ESTIMATE, not real API usage."""

from app.models import UsageInfo

CHARS_PER_TOKEN = 4  # common rough rule of thumb for English text


def estimate_tokens(text: str) -> int:
    """Estimate tokens from text length. Never negative."""
    if not text:
        return 0
    return max(0, round(len(text) / CHARS_PER_TOKEN))


class TokenTracker:
    """Adds up estimated usage across calls.

    Later you can add a method such as `record_actual(input_tokens, output_tokens)`
    that stores numbers reported by a real provider.
    """

    def __init__(self, price_per_1k_input=None, price_per_1k_output=None):
        # Prices stay None until you configure real pricing. No prices are invented.
        self.price_per_1k_input = price_per_1k_input
        self.price_per_1k_output = price_per_1k_output
        self.input_tokens = 0
        self.output_tokens = 0
        self.actual_input_tokens = None  # stays None unless a provider reports usage
        self.actual_output_tokens = None
        self.cached_input_tokens = None
        self.usage_unavailable = False

    def record_estimate(self, input_text: str, output_text: str) -> None:
        self.input_tokens += estimate_tokens(input_text)
        self.output_tokens += estimate_tokens(output_text)

    def record_actual(self, input_tokens: int, output_tokens: int, cached_tokens=None) -> None:
        """Store usage numbers reported by a real provider (not estimates)."""
        self.actual_input_tokens = (self.actual_input_tokens or 0) + input_tokens
        self.actual_output_tokens = (self.actual_output_tokens or 0) + output_tokens
        if cached_tokens is not None:
            self.cached_input_tokens = (self.cached_input_tokens or 0) + cached_tokens

    def mark_unavailable(self) -> None:
        """A real provider answered but gave no usage numbers. Do not invent any."""
        self.usage_unavailable = True

    def summary(self) -> UsageInfo:
        has_actual = self.actual_input_tokens is not None
        if has_actual:
            source = "actual"
        elif self.usage_unavailable:
            source = "unavailable"
        else:
            source = "estimated"
        cost = None
        note = "Cost is unknown: no pricing is configured."
        if source == "estimated":
            note += " Token numbers are rough estimates, not provider-reported usage."
        if source != "unavailable" and None not in (self.price_per_1k_input, self.price_per_1k_output):
            in_tokens = self.actual_input_tokens if has_actual else self.input_tokens
            out_tokens = self.actual_output_tokens if has_actual else self.output_tokens
            cost = (
                in_tokens / 1000 * self.price_per_1k_input
                + out_tokens / 1000 * self.price_per_1k_output
            )
            basis = "API-reported tokens" if has_actual else "estimated tokens"
            note = f"Cost uses user-configured prices and {basis}; it is not a bill."
        return UsageInfo(
            estimated_input_tokens=self.input_tokens,
            estimated_output_tokens=self.output_tokens,
            estimated_cost=cost,
            cost_note=note,
            is_estimate=source == "estimated",
            actual_input_tokens=self.actual_input_tokens,
            actual_output_tokens=self.actual_output_tokens,
            cached_input_tokens=self.cached_input_tokens,
            usage_source=source,
        )
