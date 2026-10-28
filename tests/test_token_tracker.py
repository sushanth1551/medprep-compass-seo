from app.token_tracker import TokenTracker, estimate_tokens


def test_estimates_are_non_negative():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("hello world") >= 0


def test_longer_text_gives_more_tokens():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_tracker_summary_is_labelled_estimate_with_unknown_cost():
    tracker = TokenTracker()
    tracker.record_estimate("input text", "output text that is longer")
    usage = tracker.summary()
    assert usage.is_estimate is True
    assert usage.estimated_input_tokens >= 0
    assert usage.estimated_output_tokens >= 0
    assert usage.estimated_cost is None  # unknown, not zero
    assert "estimate" in usage.cost_note.lower()
