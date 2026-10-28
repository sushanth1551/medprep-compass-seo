"""Lightweight source checking for the references an article lists.

What this CAN establish: a listed URL is well-formed, the page can be fetched, and (as a weak
hint) the page text contains most of the distinctive words of the claim it is cited for.
What it can NOT establish: that the source truly supports the claim, that the claim is medically
correct, or that the page is current. Those need a human. The overall result is therefore never
"verified"; the best outcome is "links_reachable".
"""

import ipaddress
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

MAX_BYTES = 300_000
TIMEOUT_SECONDS = 8.0
MAX_REDIRECTS = 3
CLAIM_MATCH_THRESHOLD = 0.6
USER_AGENT = "MedPrepCompass-SourceCheck/1.0 (+local tool; checks that cited pages exist)"
STOPWORDS = {
    "about", "after", "also", "being", "between", "could", "every", "first", "their", "there", "these",
    "those", "which", "while", "would", "should", "other", "under", "using", "where", "official",
}


@dataclass
class FetchResult:
    status: Optional[int] = None
    final_url: str = ""
    text: str = ""
    error: Optional[str] = None


def _is_public_host(host):
    """Refuse localhost, private and link-local addresses (basic protection for the server)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def default_fetch(url):
    """GET a page with a timeout, a size cap, and a few validated redirects."""
    current = url
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False,
                          headers={"User-Agent": USER_AGENT}) as client:
            for _ in range(MAX_REDIRECTS + 1):
                parts = urlparse(current)
                if parts.scheme not in ("http", "https") or not parts.hostname:
                    return FetchResult(error="Not a web address.")
                if not _is_public_host(parts.hostname):
                    return FetchResult(error="Address is not a public website; not fetched.")
                with client.stream("GET", current) as response:
                    if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("location"):
                        current = urljoin(current, response.headers["location"])
                        continue
                    body = b""
                    for chunk in response.iter_bytes():
                        body += chunk
                        if len(body) >= MAX_BYTES:
                            break
                    kind = response.headers.get("content-type", "")
                    text = body.decode(response.encoding or "utf-8", errors="replace") if "text" in kind or "html" in kind else ""
                    return FetchResult(response.status_code, current, text)
            return FetchResult(error="Too many redirects.")
    except httpx.TimeoutException:
        return FetchResult(error="Timed out.")
    except httpx.HTTPError as exc:
        return FetchResult(error=f"Could not connect ({type(exc).__name__}).")


def _page_words(html):
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def claim_overlap(claim, page_text):
    """Share of the claim's distinctive words found on the page, or None if not measurable."""
    words = {w for w in re.findall(r"[a-z0-9]{5,}", (claim or "").lower()) if w not in STOPWORDS}
    if len(words) < 3 or not page_text:
        return None
    return round(len(words & _page_words(page_text)) / len(words), 2)


def check_sources(article, fetch=default_fetch):
    """Check every listed reference. Returns a JSON-friendly dict; never claims verification."""
    results = []
    for ref in article.references:
        url = ref.get("url", "")
        item = {"title": ref.get("title", ""), "url": url, "claim": ref.get("claim", ""),
                "reachable": False, "http_status": None, "final_url": "", "claim_word_overlap": None,
                "support": "not_established", "note": ""}
        parts = urlparse(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            item["note"] = "Not a valid web address."
        else:
            got = fetch(url)
            item["http_status"], item["final_url"] = got.status, got.final_url
            if got.error:
                item["note"] = got.error
            elif got.status and 200 <= got.status < 300:
                item["reachable"] = True
                overlap = claim_overlap(item["claim"], got.text)
                item["claim_word_overlap"] = overlap
                if overlap is None:
                    item["note"] = "Page exists; the claim could not be compared automatically."
                elif overlap >= CLAIM_MATCH_THRESHOLD:
                    item["support"] = "words_found"
                    item["note"] = "Most claim words appear on the page. This is a hint, not proof of support."
                else:
                    item["support"] = "words_missing"
                    item["note"] = "Few claim words appear on the page. Check the source by hand."
            else:
                item["note"] = f"The page answered with HTTP {got.status}."
        results.append(item)
    if not results:
        overall = "no_sources"
    elif all(r["reachable"] for r in results):
        overall = "links_reachable"
    else:
        overall = "incomplete"
    return {
        "performed": True,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": overall,
        "sources": results,
        "limits": "Reachable links are not verified facts. A human must read each source and judge the claim.",
    }
