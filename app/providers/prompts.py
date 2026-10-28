"""Prompt text for the JSON-writing providers.

Styles:
- "baseline": the Phase 2 wording, kept unchanged for benchmark comparisons.
- "optimized": the Phase 3 shortened wording (same output fields as baseline).
- "structured": the default. Same rules, plus search intent, audience, references with the
  claim each supports, and review flags, so the quality workflow has data to check.
"""

PROMPT_STYLES = ("structured", "optimized", "baseline")
DEFAULT_PROMPT_STYLE = "structured"

BASELINE_SYSTEM = (
    "You write draft blog articles for a medical-exam preparation website. "
    "Rules: never invent citations, URLs, statistics, or study names. Do not promise "
    "exam success or scores. Avoid specific medical facts you are not certain of; keep "
    "advice general and about study strategy. Use the keyword naturally, never stuff it; "
    "include it in the title and meta description. "
    "Reply with ONE JSON object only, no markdown fences, in exactly this shape: "
    '{"title": str, "meta_description": str (max 160 chars), '
    '"sections": [{"heading": str, "body": str}], '
    '"faq": [{"question": str, "answer": str}]}'
)

OPTIMIZED_SYSTEM = (
    "Write a draft blog article for a medical-exam prep site; humans will review it. "
    "Never invent citations, URLs, statistics or study names. No promises of exam "
    "success. Keep to general study-strategy advice; skip medical facts you are unsure of. "
    "Use the keyword naturally, including in the title and meta description. Output only JSON: "
    '{"title":str,"meta_description":str(<=160 chars),'
    '"sections":[{"heading":str,"body":str}],"faq":[{"question":str,"answer":str}]}'
)

STRUCTURED_SYSTEM = (
    "Write a draft blog article for a medical-exam prep site; humans will review it. "
    "Never invent citations, URLs, statistics, quotations or study names. No promises of exam "
    "success. Keep to general study-strategy advice; skip medical facts you are unsure of. "
    "Use the keyword naturally, including in the title and meta description; no filler or "
    "repetition. Write EVERY listed section, in order, with full text (the conclusion is the last "
    "section, never the only one) and send each key only once. You may add '### ' sub-headings "
    "inside a body (never '####' or deeper). "
    "List a reference only if you are certain the official page exists (for example usmle.org or "
    "nbme.org) and state the claim it supports; otherwise leave references empty and put "
    "uncertain claims in review_flags. Output only JSON: "
    '{"title":str,"meta_description":str(<=160 chars),"search_intent":str,"audience":str,'
    '"sections":[{"heading":str,"body":str}],"faq":[{"question":str,"answer":str}],'
    '"references":[{"title":str,"url":str,"claim":str}],"review_flags":[str]}'
)

CONTINUE_SYSTEM = (
    "You finish a draft blog article for a medical-exam prep site. Write ONLY the missing sections "
    "listed by the user; do not repeat the existing ones. Never invent citations, URLs, statistics "
    "or study names. No promises of exam success. Output only JSON: "
    '{"sections":[{"heading":str,"body":str}]}'
)

REGEN_SYSTEM = (
    "Write one section of a draft blog article for a medical-exam prep site. "
    "Never invent citations, statistics or study names; no promises of exam success. "
    'Output only JSON: {"body": str}'
)

# Article-length choices offered by the web form: (target words, label). A target, not a guarantee.
ARTICLE_LENGTHS = {
    "short": (550, "about 500-600 words"),
    "medium": (900, "about 900 words"),
    "long": (1500, "about 1,500 words"),
}


def system_prompt(style):
    if style == "baseline":
        return BASELINE_SYSTEM
    return OPTIMIZED_SYSTEM if style == "optimized" else STRUCTURED_SYSTEM


def _length_line(target_words):
    if not target_words:
        return ""
    low, high = int(target_words * 0.9), int(target_words * 1.1)
    return (
        f" Target length: about {target_words} words in total (aim for {low}-{high}), "
        f"roughly {target_words // 6} words per section."
    )


def user_prompt(style, topic, keyword, outline, target_words=None):
    if style == "baseline":
        headings = "\n".join(f"- {o.heading}: {o.purpose}" for o in outline)
        return (
            f"Topic: {topic}\nTarget keyword: {keyword}\n"
            f"Use these section headings in order:\n{headings}\n"
            "Add 3 to 5 FAQ items. Write a draft for human review." + _length_line(target_words)
        )
    headings = " | ".join(o.heading for o in outline)
    count = f" Write all {len(outline)} sections." if style == "structured" else ""
    return (
        f"Topic: {topic}\nKeyword: {keyword}\nSection headings in order: {headings}\n"
        "FAQ: 3 to 5 items." + count + _length_line(target_words)
    )


def continue_prompt(topic, keyword, written, missing, words_each=None):
    size = f" about {words_each} words each," if words_each else ""
    return (
        f"Topic: {topic}\nKeyword: {keyword}\nAlready written (do not repeat): {' | '.join(written)}\n"
        f"Write these missing sections in order,{size} using exactly these headings: {' | '.join(missing)}"
    )


def regen_prompt(topic, keyword, heading):
    return f'Topic: {topic}\nKeyword: {keyword}\nWrite only the section "{heading}".'
