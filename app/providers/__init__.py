"""Providers turn an outline into article content.

A provider is any object with:
    name: str
    generate_sections(topic, keyword, outline) -> list[(heading, body)]
    generate_faq(topic, keyword) -> list[(question, answer)]
A real Claude provider can follow the same shape later.
"""

from app.providers.mock_provider import MockProvider

__all__ = ["MockProvider"]
