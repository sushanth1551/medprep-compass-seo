"""Mock provider: offline, deterministic, clearly labelled demo text.

This is NOT the Claude API. It makes no network calls and verifies no facts.
"""

from app.models import DEMO_NOTICE
from app.providers.prompts import DEFAULT_PROMPT_STYLE, system_prompt, user_prompt


class MockProvider:
    name = "MockProvider (demo only, not a real AI model)"
    provider_id = "mock"
    model = None  # no model is involved
    call_count = 0  # the mock never calls a model

    def __init__(self, prompt_style=DEFAULT_PROMPT_STYLE, target_words=None):
        self.prompt_style = prompt_style
        self.target_words = target_words

    def build_prompt_text(self, topic, keyword, outline):
        """The text a real model WOULD receive; used only for token ESTIMATES."""
        return system_prompt(self.prompt_style) + "\n" + user_prompt(
            self.prompt_style, topic, keyword, outline, self.target_words
        )

    def regenerate_section(self, topic, keyword, heading):
        """Offline stand-in used for testing the revision flow; makes no model call."""
        return (
            f"[DEMO PLACEHOLDER] Rewritten section '{heading}' about {topic}. "
            f"Keyword to mention naturally: {keyword}. {DEMO_NOTICE}"
        )

    def generate_sections(self, topic, keyword, outline):
        sections = []
        for item in outline:
            body = (
                f"[DEMO PLACEHOLDER] This section would cover the following: {item.purpose}. "
                f"Topic: \"{topic}\". Keyword to mention naturally: \"{keyword}\". "
                "No medical facts or citations are included here. "
                f"{DEMO_NOTICE}"
            )
            sections.append((item.heading, body))
        return sections

    def generate_faq(self, topic, keyword):
        return [
            (
                f"What is {keyword}?",
                f"[DEMO PLACEHOLDER] A short answer about {keyword} would go here. {DEMO_NOTICE}",
            ),
            (
                f"How should students approach {topic}?",
                f"[DEMO PLACEHOLDER] A study-approach answer would go here. {DEMO_NOTICE}",
            ),
            (
                f"Where can I find reliable sources on {topic}?",
                f"[DEMO PLACEHOLDER] Recommend official exam bodies and textbooks here, "
                f"after an editor verifies them. {DEMO_NOTICE}",
            ),
        ]
