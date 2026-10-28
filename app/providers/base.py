"""Shared logic for providers that ask a chat model for a JSON article.

Subclasses only implement `_complete` (one API request). Everything else - prompts,
JSON parsing, truncation checks, bounded continuation and call/usage bookkeeping - lives here.

Paid requests are bounded: one request for the article, plus at most `max_continuations`
(default 1) to write sections the first reply left out. There are no other retries.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from app.models import DEFAULT_MAX_OUTPUT_TOKENS, DRAFT_NOTICE, ProviderError
from app.providers.prompts import (
    CONTINUE_SYSTEM,
    DEFAULT_PROMPT_STYLE,
    PROMPT_STYLES,
    REGEN_SYSTEM,
    continue_prompt,
    regen_prompt,
    system_prompt,
    user_prompt,
)

DEFAULT_MAX_CONTINUATIONS = 1
MAX_RAW_CHARS = 30000  # how much of a suspicious reply is kept in the diagnostics file


@dataclass
class Completion:
    text: str
    input_tokens: Optional[int] = None  # None = the API did not report it
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    truncated: bool = False
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class JsonChatProvider:
    provider_id = ""
    label = ""
    notice = DRAFT_NOTICE

    def __init__(
        self, model, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS, prompt_style=DEFAULT_PROMPT_STYLE,
        target_words=None, max_continuations=DEFAULT_MAX_CONTINUATIONS,
    ):
        if not model:
            raise ProviderError(f"{self.label} needs a model name.")
        if prompt_style not in PROMPT_STYLES:
            raise ProviderError(f"Unknown prompt style '{prompt_style}'.")
        self.model = model
        self.max_tokens = max_tokens
        self.prompt_style = prompt_style
        self.target_words = target_words
        self.max_continuations = max(0, int(max_continuations))
        self.continuations_used = 0
        self.name = f"{self.label} (model: {model})"
        self.call_count = 0
        self.call_log = []
        self.diagnostics = []  # facts about unusual replies; saved by the workflow, never secrets
        self._cache = None

    def _complete(self, system, user):
        raise NotImplementedError

    # Bookkeeping --------------------------------------------------------------
    def _request(self, system, user):
        self.call_count += 1  # a failed request still counts as an attempt
        completion = self._complete(system, user)
        self.call_log.append(
            {
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "cached_tokens": completion.cached_tokens,
                "truncated": completion.truncated,
                "reasoning_tokens": completion.reasoning_tokens,
                "finish_reason": completion.finish_reason,
            }
        )
        if completion.truncated:
            self._note("output_limit_reached", completion, planned=None)
            used = ""
            if completion.output_tokens is not None:
                used = f" The model produced {completion.output_tokens} output tokens"
                if completion.reasoning_tokens:
                    used += f", {completion.reasoning_tokens} of them reasoning"
                used += "."
            raise ProviderError(
                f"{self.label} reply was cut off: the output limit of {self.max_tokens} tokens was "
                f"reached.{used} No draft was saved and nothing was retried. Raise "
                "MAX_OUTPUT_TOKENS in .env (for example 8000), restart the app, and try again."
            )
        return completion

    def _note(self, event, completion, planned, **facts):
        self.diagnostics.append({
            "event": event,
            "call_number": self.call_count,
            "finish_reason": completion.finish_reason,
            "output_tokens": completion.output_tokens,
            "reasoning_tokens": completion.reasoning_tokens,
            "planned_sections": planned,
            "reply_chars": len(completion.text or ""),
            "raw_reply": (completion.text or "")[:MAX_RAW_CHARS],  # the model's article text only
            **facts,
        })

    def usage_totals(self):
        """Summed API-reported usage, or None if any call lacked usage numbers."""
        if not self.call_log or any(
            c["input_tokens"] is None or c["output_tokens"] is None for c in self.call_log
        ):
            return None
        cached = [c["cached_tokens"] for c in self.call_log if c["cached_tokens"] is not None]
        return {
            "input_tokens": sum(c["input_tokens"] for c in self.call_log),
            "output_tokens": sum(c["output_tokens"] for c in self.call_log),
            "cached_input_tokens": sum(cached) if cached else None,
        }

    # Provider interface used by the workflow -------------------------------------
    def build_prompt_text(self, topic, keyword, outline):
        return system_prompt(self.prompt_style) + "\n" + user_prompt(
            self.prompt_style, topic, keyword, outline, self.target_words
        )

    def generate_metadata(self, topic, keyword, outline):
        data = self._generate(topic, keyword, outline)
        return data["title"], data["meta_description"][:160]

    def generate_sections(self, topic, keyword, outline):
        return list(self._generate(topic, keyword, outline)["sections"])

    def extras(self):
        """Optional structured fields from the last reply (empty for older prompt styles)."""
        data = (self._cache or {}).get("data", {})
        return {k: data.get(k) for k in ("search_intent", "audience", "references", "review_flags")}

    def generate_faq(self, topic, keyword):
        if self._cache is None:
            raise ProviderError("generate_faq was called before generate_sections.")
        return list(self._cache["data"]["faq"])

    def regenerate_section(self, topic, keyword, heading):
        """Rewrite ONE section body. Used only when explicitly enabled."""
        completion = self._request(REGEN_SYSTEM, regen_prompt(topic, keyword, heading))
        try:
            body = str(json.loads(_strip_fences(completion.text))["body"]).strip()
        except (ValueError, KeyError, TypeError):
            raise ProviderError("Section rewrite was not in the expected JSON form.") from None
        if not body:
            raise ProviderError("Section rewrite came back empty.")
        return body

    def _generate(self, topic, keyword, outline):
        key = (topic, keyword)
        if self._cache and self._cache["key"] == key:
            return self._cache["data"]
        completion = self._request(
            system_prompt(self.prompt_style),
            user_prompt(self.prompt_style, topic, keyword, outline, self.target_words),
        )
        planned = len(outline)
        try:
            data = self._parse(completion.text)
        except ProviderError:
            self._note("unparseable_reply", completion, planned)
            raise
        info = data.pop("_parse_info")
        missing = missing_headings(outline, data["sections"])
        if info["duplicate_keys"] or missing:
            self._note(
                "sections_missing" if missing else "repeated_json_keys", completion, planned,
                duplicate_keys=info["duplicate_keys"], sections_in_reply=len(data["sections"]),
                missing_headings=[o.heading for o in missing],
            )
        if missing and self.continuations_used < self.max_continuations:
            data["sections"] = self._continue(topic, keyword, outline, data["sections"], missing)
        got = len(data["sections"])
        if planned and got * 2 < planned:
            tried = " One follow-up request to fill the gaps did not help." if self.continuations_used else ""
            # The reply finished normally (so this is NOT an output-limit problem) but most of the
            # article is missing. Saving it as a draft would be misleading.
            raise ProviderError(
                f"The model returned only {got} of {planned} requested sections, so the article is "
                "unusable. The reply finished normally, so this is not a token-limit problem. No draft "
                f"was saved.{tried} Try generating again, or try a different model. "
                "(Diagnostics were saved under reports/diagnostics.)"
            )
        self._cache = {"key": key, "data": data}
        return data

    def _continue(self, topic, keyword, outline, existing, missing):
        """ONE extra request that writes only the missing sections. Visible in the call count."""
        self.continuations_used += 1
        per_section = (self.target_words or 0) // max(1, len(outline)) or None
        completion = self._request(
            CONTINUE_SYSTEM,
            continue_prompt(topic, keyword, [h for h, _ in existing], [o.heading for o in missing], per_section),
        )
        try:
            raw = json.loads(_strip_fences(completion.text), object_pairs_hook=_merge_repeated_keys)
            written = [(str(s["heading"]).strip(), str(s["body"]).strip()) for s in raw["sections"]]
        except (ValueError, KeyError, TypeError):
            self._note("continuation_unparseable", completion, len(outline))
            return existing  # keep what we have; the completeness checks will flag the rest
        return merge_sections(outline, existing, written)

    @staticmethod
    def _parse(text):
        try:
            raw = json.loads(_strip_fences(text), object_pairs_hook=_merge_repeated_keys)
            title = str(raw["title"]).strip()
            meta = str(raw["meta_description"]).strip()
            sections = [(str(s["heading"]).strip(), str(s["body"]).strip()) for s in raw["sections"]]
            faq = [(str(f["question"]).strip(), str(f["answer"]).strip()) for f in raw["faq"]]
            extras = _optional_fields(raw)
        except (ValueError, KeyError, TypeError):
            raise ProviderError(
                "The model's reply was not in the expected JSON structure (it may have been "
                "cut off); no draft saved."
            ) from None
        if not title or not sections:
            raise ProviderError("The model's reply had no title or sections; no draft saved.")
        return {
            "title": title, "meta_description": meta, "sections": _dedupe(sections), "faq": faq,
            **extras, "_parse_info": {"duplicate_keys": list(raw.get("__repeated_keys__", []))},
        }


# Parsing helpers ------------------------------------------------------------------
def _merge_repeated_keys(pairs):
    """JSON objects with a repeated key normally keep only the LAST value, silently losing the
    earlier ones. Models sometimes repeat "sections"; join those lists instead of dropping them."""
    out, repeated = {}, []
    for key, value in pairs:
        if key in out:
            repeated.append(key)
            if isinstance(out[key], list) and isinstance(value, list):
                out[key] = out[key] + value
                continue
        out[key] = value
    if repeated:
        out["__repeated_keys__"] = repeated
    return out


def _norm(text):
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _dedupe(sections):
    """Keep the first copy of any heading that appears twice."""
    seen, kept = set(), []
    for heading, body in sections:
        if _norm(heading) not in seen:
            seen.add(_norm(heading))
            kept.append((heading, body))
    return kept


def missing_headings(outline, sections):
    """Planned sections the reply did not contain. If no heading matches at all (the model renamed
    them), assume the reply covers the first N planned sections."""
    have = {_norm(h) for h, _ in sections}
    matched = [o for o in outline if _norm(o.heading) in have]
    if not matched:
        return list(outline[len(sections):])
    return [o for o in outline if _norm(o.heading) not in have]


def merge_sections(outline, existing, written):
    """Combine the first reply with the continuation: outline order, no duplicates."""
    by_heading = {}
    for heading, body in list(existing) + list(written):
        by_heading.setdefault(_norm(heading), (heading, body))  # existing text wins over new text
    wanted = {_norm(o.heading) for o in outline}
    if not any(_norm(h) in wanted for h, _ in existing):  # renamed headings: keep reply order
        extra = [(h, b) for h, b in written if _norm(h) not in {_norm(x) for x, _ in existing}]
        return _dedupe(list(existing) + extra)
    ordered = [by_heading[_norm(o.heading)] for o in outline if _norm(o.heading) in by_heading]
    leftovers = [v for k, v in by_heading.items() if k not in wanted]
    return ordered + leftovers


def _optional_fields(raw):
    """Read the optional fields tolerantly: a bad optional field never fails the article."""
    refs = []
    for ref in raw.get("references") or []:
        if isinstance(ref, dict) and str(ref.get("url", "")).strip():
            refs.append({
                "title": str(ref.get("title", "")).strip()[:200],
                "url": str(ref["url"]).strip()[:500],
                "claim": str(ref.get("claim", "")).strip()[:400],
            })
    flags = [str(f).strip()[:300] for f in (raw.get("review_flags") or []) if str(f).strip()]
    return {
        "search_intent": str(raw.get("search_intent") or "").strip()[:300],
        "audience": str(raw.get("audience") or "").strip()[:300],
        "references": refs,
        "review_flags": flags,
    }


def _strip_fences(text):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
