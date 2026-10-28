"""Deterministic SEO and review checks. No AI call, no internet.

These are STRUCTURAL checks. They cannot tell whether a medical statement is true
or whether a source exists. Flagged sentences are only prompts for a human reviewer.
"""

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from app.models import SeoReport

MIN_WORDS = 300
MAX_WORDS = 2500
MAX_META_CHARS = 160
MAX_AVG_SENTENCE_WORDS = 25


@dataclass
class SeoConfig:
    """Thresholds for the extra checks. Change them to suit your site."""

    title_min_chars: int = 30
    title_max_chars: int = 65
    meta_min_chars: int = 70
    faq_min_answer_words: int = 12
    length_tolerance: float = 0.25  # allowed distance from the selected target length
    max_sentence_words: int = 40  # warn when the longest sentence exceeds this
KEYWORD_DENSITY_WARN = 0.03  # warn above 3 %, to discourage keyword stuffing

# Sentences matching these are flagged as "needs evidence", nothing more.
CLAIM_PATTERNS = [
    r"\d+(\.\d+)?\s*%",
    r"\b(studies|research|evidence|trials?|data|surveys?)\s+(show|shows|suggest|suggests|indicate|indicates|demonstrate|demonstrates|found)\b",
    r"\baccording to\b",
    r"\b(proven|clinically proven|always|never)\b",
    r"\b(first-line|contraindicated|mortality|incidence|prevalence|sensitivity|specificity)\b",
    r"\b\d+\s?(mg|mcg|ml|mmhg|mmol)\b",
    r"\bpass rates?\b",
]
# Promises about exam outcomes are not allowed in the article at all.
PROMISE_PATTERN = re.compile(
    r"\bguarantee[sd]?\b|\byou (will|'ll) (pass|score|succeed)\b|\bensures? (you )?(pass|success)\b"
    r"|\bsure[- ]fire\b|\b100\s?% (pass|success)\b",
    re.IGNORECASE,
)
REFERENCE_PATTERNS = [
    r"https?://\S+",
    r"\([A-Z][A-Za-z\-]+(?: et al\.)?,? \d{4}\)",
    r"\[\d+\]",
    r"\bPMID:?\s?\d+",
    r"\bdoi:\s?\S+",
]


NEGATION = re.compile(r"\b(not|no|never|cannot|can't|won't|doesn't|don't|isn't|without|nor)\b", re.I)


def _has_promise(text):
    """True if the text promises exam success. 'We do not guarantee a pass' is NOT a promise."""
    for match in PROMISE_PATTERN.finditer(text):
        before = text[max(0, match.start() - 40):match.start()]
        sentence_start = max(before.rfind("."), before.rfind("!"), before.rfind("?"))
        if not NEGATION.search(before[sentence_start + 1:]):
            return True
    return False


def _heading_level_problems(article):
    """Sections are H2. Inside a body, a heading may go at most one level deeper than the last."""
    for _, body in article.sections:
        current = 2
        for hashes in re.findall(r"^(#+)\s", body, re.M):
            level = len(hashes)
            if level < 2 or level > current + 1:
                return (f"Heading levels skip or break the hierarchy (found H{level} after H{current}); "
                        "use H3 under an H2 section, and H4 only under an H3.")
            current = level
    return ""


def _normalise(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _body_text(article):
    parts = [body for _, body in article.sections] + [ans for _, ans in article.faq]
    return "\n".join(parts)


def count_words(article):
    """The one word count used everywhere (SEO checks, completeness, reports): sections + FAQ."""
    return len(re.findall(r"\b\w+\b", _body_text(article)))


def find_claims_needing_evidence(article):
    flagged = []
    for sentence in _sentences(_body_text(article)):
        if any(re.search(p, sentence, re.IGNORECASE) for p in CLAIM_PATTERNS):
            flagged.append(sentence)
    return flagged


def find_references(article):
    found = []
    text = article.to_markdown()
    for pattern in REFERENCE_PATTERNS:
        found += [m.group(0) for m in re.finditer(pattern, text)]
    return found


def run_seo_checks(article, config=None, target_words=None):
    report = SeoReport()
    cfg = config or SeoConfig()

    def record(level, message):
        getattr(report, level).append(message)

    # Title and meta description
    if article.title.strip():
        record("passed", "Title is present.")
    else:
        record("failed", "Title is empty.")
    if article.meta_description.strip():
        record("passed", "Meta description is present.")
    else:
        record("failed", "Meta description is empty.")
    if len(article.meta_description) > MAX_META_CHARS:
        record("failed", f"Meta description is longer than {MAX_META_CHARS} characters.")
    elif article.meta_description.strip():
        record("passed", f"Meta description is within {MAX_META_CHARS} characters.")

    # Keyword placement (a warning only: forcing it in would read unnaturally)
    keyword = _normalise(article.keyword)
    if keyword and keyword in _normalise(article.title):
        record("passed", "Target keyword appears in the title.")
    else:
        record("warnings", "Target keyword does not appear in the title.")
    if keyword and keyword in _normalise(article.meta_description):
        record("passed", "Target keyword appears in the meta description.")
    else:
        record("warnings", "Target keyword does not appear in the meta description.")

    # Headings and sections
    if len(article.sections) >= 3:
        record("passed", f"Article has {len(article.sections)} headed sections (3 or more).")
    else:
        record("failed", "Article has fewer than 3 headed sections.")
    headings = [_normalise(h) for h, _ in article.sections]
    if any(not h for h in headings) or len(set(headings)) != len(headings):
        record("failed", "Some headings are empty or duplicated.")
    else:
        record("passed", "Headings are non-empty and unique.")
    if any(not body.strip() for _, body in article.sections):
        record("failed", "At least one section has no body text.")
    else:
        record("passed", "Every section has body text.")

    # FAQ
    if article.faq:
        record("passed", f"FAQ section present ({len(article.faq)} questions).")
    else:
        record("failed", "FAQ section is missing.")

    # Length
    words = re.findall(r"\b\w+\b", _body_text(article))
    report.word_count = len(words)
    if len(words) < MIN_WORDS:
        record("warnings", f"Body is about {len(words)} words; under {MIN_WORDS} may be thin.")
    elif len(words) > MAX_WORDS:
        record("warnings", f"Body is about {len(words)} words; over {MAX_WORDS} may be too long.")
    else:
        record("passed", f"Body length is about {len(words)} words.")

    # Keyword density (stuffing guard)
    if keyword and words:
        hits = _normalise(_body_text(article)).count(keyword)
        density = hits * len(keyword.split()) / len(words)
        if density > KEYWORD_DENSITY_WARN:
            record("warnings", f"Keyword makes up {density:.1%} of the body; this may read as stuffing.")
        else:
            record("passed", "Keyword use does not look excessive.")

    # Readability indicators
    sentences = _sentences(_body_text(article))
    if sentences:
        lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        avg = sum(lengths) / len(lengths)
        report.readability = {
            "sentences": len(sentences),
            "average_sentence_words": round(avg, 1),
            "longest_sentence_words": max(lengths),
        }
        if avg > MAX_AVG_SENTENCE_WORDS:
            record("warnings", f"Average sentence is {avg:.1f} words; consider shorter sentences.")
        else:
            record("passed", f"Average sentence length is {avg:.1f} words.")

    # Promises and claims
    if _has_promise(article.to_markdown()):
        record("failed", "Text appears to promise exam success. Remove unsupported guarantees.")
    else:
        record("passed", "No exam-success promises detected.")

    report.references_found = find_references(article)
    report.claims_needing_evidence = find_claims_needing_evidence(article)
    if report.references_found:
        record(
            "warnings",
            f"{len(report.references_found)} source reference(s) detected. They have NOT been "
            "verified; a human must confirm each one exists and supports the claim.",
        )
    elif report.claims_needing_evidence:
        record("warnings", "Factual-sounding claims were found but no source references.")
    else:
        record("passed", "No factual-sounding claims were flagged (this is not fact verification).")
    if report.claims_needing_evidence:
        record(
            "warnings",
            f"{len(report.claims_needing_evidence)} sentence(s) look like factual claims that need evidence.",
        )
    _extra_checks(article, report, cfg, target_words)
    return report


def _extra_checks(article, report, cfg, target_words):
    """Phase 5 checks. Every result is a plain pass, warning or failure; nothing is scored."""
    record = lambda level, message: getattr(report, level).append(message)  # noqa: E731

    n = len(article.title.strip())
    if n:
        if cfg.title_min_chars <= n <= cfg.title_max_chars:
            record("passed", f"Title length is {n} characters (suggested {cfg.title_min_chars}-{cfg.title_max_chars}).")
        else:
            record("warnings", f"Title is {n} characters; {cfg.title_min_chars}-{cfg.title_max_chars} is the suggested range.")
    m = len(article.meta_description.strip())
    if 0 < m < cfg.meta_min_chars:
        record("warnings", f"Meta description is only {m} characters; under {cfg.meta_min_chars} may be too thin.")

    # Heading hierarchy: sections are H2; a body may use H3 only. H1 is the title.
    problems = _heading_level_problems(article)
    if problems:
        record("failed", problems)
    elif any(re.search(r"^#{3,}\s", body, re.M) for _, body in article.sections):
        record("passed", "Sub-headings inside sections never skip a level (H3 under H2).")
    if len(article.sections) >= 2 and len({_normalise(h) for h, _ in article.sections}) == len(article.sections):
        generic = [h for h, _ in article.sections if _normalise(h) in {"section", "heading", "untitled", "text"}]
        if generic:
            record("warnings", "Some headings are generic and not meaningful.")

    # FAQ usefulness
    thin = [q for q, a in article.faq if len(re.findall(r"\b\w+\b", a)) < cfg.faq_min_answer_words]
    if article.faq and thin:
        record("warnings", f"{len(thin)} FAQ answer(s) are shorter than {cfg.faq_min_answer_words} words.")
    elif article.faq:
        record("passed", "FAQ answers have a useful length.")

    # Length relative to the selected target
    if target_words:
        low, high = target_words * (1 - cfg.length_tolerance), target_words * (1 + cfg.length_tolerance)
        if report.word_count < low:
            record("warnings", f"Body is {report.word_count} words, under the {int(low)}-word floor for a {target_words}-word target.")
        elif report.word_count > high:
            record("warnings", f"Body is {report.word_count} words, over the {int(high)}-word ceiling for a {target_words}-word target.")
        else:
            record("passed", f"Body length ({report.word_count} words) is near the {target_words}-word target.")

    if report.readability and report.readability.get("longest_sentence_words", 0) > cfg.max_sentence_words:
        record("warnings", f"The longest sentence has {report.readability['longest_sentence_words']} words.")

    # Reference URLs: syntax only. Whether they exist is a separate source-verification step.
    for ref in article.references:
        url = urlparse(ref.get("url", ""))
        if url.scheme not in ("http", "https") or not url.netloc:
            record("failed", f"Reference URL is not a valid web address: {ref.get('url', '')[:80]!r}.")
    if article.references and not any("valid web address" in f for f in report.failed):
        record("passed", f"{len(article.references)} reference URL(s) are well-formed (existence NOT checked here).")
