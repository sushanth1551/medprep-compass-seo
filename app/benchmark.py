"""Repeatable benchmark: baseline prompt vs optimized prompt on fixed topics.

    python -m app.benchmark                                  # mock: free, offline
    python -m app.benchmark --live --i-understand-this-spends-credits

Mock results use character-based token ESTIMATES and prove nothing about real savings.
A live run makes one request per topic per prompt style (6 with the default topics) and
stops at the first failure so it cannot keep spending after an error.
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import build_provider, get_provider_name, load_local_env
from app.models import ConfigError, ProviderError
from app.providers.mock_provider import MockProvider
from app.workflow import run_workflow

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "reports" / "benchmarks"
TOPICS = [
    ("How to Build a USMLE Step 1 Study Schedule", "usmle step 1 study schedule"),
    ("Pharmacology Revision Strategies for Medical Students", "pharmacology revision strategies"),
    ("How to Review Practice Questions Effectively", "review practice questions"),
]
STYLES = ("baseline", "optimized")
MIN_WORD_RATIO = 0.8  # optimized article must keep at least 80 % of the baseline length


def _total(m):
    if m.get("input_tokens") is None or m.get("output_tokens") is None:
        return None
    return m["input_tokens"] + m["output_tokens"]


def _pct(before, after):
    if before in (None, 0) or after is None:
        return None
    return round((after - before) / before * 100, 1)


def compare(baseline, optimized, mock):
    """Compare two runs of one topic. Percentages only when the comparison is valid."""
    out = {"topic": baseline["topic"], "valid": False, "percent_change": None, "notes": []}
    if baseline["status"] != "ok" or optimized["status"] != "ok":
        out["notes"].append("A run failed, so there is nothing to compare.")
        return out
    if (baseline["provider"], baseline["model"]) != (optimized["provider"], optimized["model"]):
        out["notes"].append("Different provider/model: not comparable.")
        return out
    if baseline["usage_source"] != optimized["usage_source"] or baseline["usage_source"] not in (
        "actual", "estimated"
    ):
        out["notes"].append("Token numbers are not of the same kind (or unavailable): not comparable.")
        return out
    if not (baseline["appears_complete"] and optimized["appears_complete"]):
        out["notes"].append("An article looks incomplete: token change is not counted as a saving.")
        return out
    if optimized["word_count"] < MIN_WORD_RATIO * baseline["word_count"]:
        out["notes"].append(
            "Optimized article is much shorter than the baseline: lower tokens may just mean "
            "less content, so this is not counted as a saving."
        )
        return out
    if optimized["seo"]["failed"] > baseline["seo"]["failed"]:
        out["notes"].append("Optimized article fails more SEO checks: not counted as a saving.")
        return out
    out["valid"] = True
    out["percent_change"] = {
        "input_tokens": _pct(baseline["input_tokens"], optimized["input_tokens"]),
        "output_tokens": _pct(baseline["output_tokens"], optimized["output_tokens"]),
        "total_tokens": _pct(_total(baseline), _total(optimized)),
        "model_calls": _pct(baseline["model_calls"], optimized["model_calls"]),
        "elapsed_seconds": None if mock else _pct(
            baseline["elapsed_seconds"], optimized["elapsed_seconds"]
        ),
    }
    if mock:
        out["notes"].append("MOCK ESTIMATE: says nothing about real token savings.")
    return out


def run_benchmark(mode="mock", env=None, out_dir=BENCHMARK_DIR, topics=TOPICS):
    """Run every topic with both prompt styles and write JSON + Markdown reports."""
    mock = mode == "mock"
    if mock:
        make_provider = lambda style: MockProvider(prompt_style=style)  # noqa: E731
    else:
        if get_provider_name(env) == "mock":
            raise ConfigError("Live benchmark needs AI_PROVIDER=claude or groq in your .env.")
        make_provider = lambda style: build_provider(env, prompt_style=style)  # noqa: E731

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stem = f"benchmark-{mode}-{stamp}-{uuid.uuid4().hex[:6]}"
    out_dir = Path(out_dir)
    work_dir = out_dir / stem
    runs, aborted = [], None
    for topic, keyword in topics:
        for style in STYLES:
            try:
                result = run_workflow(
                    topic, keyword, provider=make_provider(style),
                    output_dir=work_dir / "drafts", metrics_dir=work_dir / "runs",
                )
            except ProviderError as exc:
                aborted = f"Stopped at '{topic}' ({style}): {exc}"
                break
            runs.append(result.metrics)
        if aborted:
            break

    comparisons = []
    for topic, _ in topics:
        by_style = {r["prompt_style"]: r for r in runs if r["topic"] == topic}
        if len(by_style) == 2:
            comparisons.append(compare(by_style["baseline"], by_style["optimized"], mock))
    report = {
        "mode": mode,
        "label": "MOCK ESTIMATES - not evidence of real savings" if mock else "LIVE API RUN",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aborted": aborted,
        "runs": runs,
        "comparisons": comparisons,
        "caveats": [
            "Automated SEO checks cannot guarantee search rankings or factual correctness.",
            "Human review is required for every article.",
            "One run per topic/style is a small sample; model output varies between runs.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return report, json_path, md_path


def _fmt(value):
    return "n/a" if value is None else str(value)


def render_markdown(report):
    lines = [f"# Benchmark ({report['mode']})", "", f"**{report['label']}**", ""]
    if report["aborted"]:
        lines += [f"**ABORTED:** {report['aborted']}", ""]
    lines += [
        "| Topic | Prompt | Provider / model | Tokens are | In | Out | Total | Calls | Secs | Words "
        "| Complete | SEO pass/warn/fail | Claims to check | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["runs"]:
        cost = r["cost"]
        cost_text = "unknown" if cost["amount"] is None else f"{cost['amount']:.6f} {cost['currency']}"
        seo = r["seo"]
        lines.append(
            f"| {r['topic']} | {r['prompt_style']} | {r['provider']} / {_fmt(r['model'])} "
            f"| {r['usage_source']} | {_fmt(r['input_tokens'])} | {_fmt(r['output_tokens'])} "
            f"| {_fmt(_total(r))} | {r['model_calls']} | {r['elapsed_seconds']} | {r['word_count']} "
            f"| {'yes' if r['appears_complete'] else 'NO'} "
            f"| {seo['passed']}/{seo['warnings']}/{seo['failed']} "
            f"| {r['human_review_flags']['claims_needing_evidence']} | {cost_text} |"
        )
    lines += ["", "## Baseline vs optimized", ""]
    for c in report["comparisons"]:
        if c["valid"]:
            p = c["percent_change"]
            lines.append(
                f"- **{c['topic']}**: input {_fmt(p['input_tokens'])}%, output "
                f"{_fmt(p['output_tokens'])}%, total {_fmt(p['total_tokens'])}%"
                + (f", calls {p['model_calls']}%" if p["model_calls"] is not None else "")
                + " (negative = fewer). " + " ".join(c["notes"])
            )
        else:
            lines.append(f"- **{c['topic']}**: no valid comparison. " + " ".join(c["notes"]))
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in report["caveats"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Baseline vs optimized prompt benchmark.")
    parser.add_argument("--live", action="store_true", help="use the real provider from .env")
    parser.add_argument("--i-understand-this-spends-credits", action="store_true")
    args = parser.parse_args(argv)
    if args.live and not args.i_understand_this_spends_credits:
        print(
            "Live benchmark makes real, paid API requests (one per topic per prompt style).\n"
            "Re-run with --i-understand-this-spends-credits to confirm."
        )
        return 2
    load_local_env()
    try:
        report, json_path, md_path = run_benchmark("live" if args.live else "mock")
    except ConfigError as exc:
        print(f"Configuration problem: {exc}")
        return 2
    print(report["label"])
    print("JSON:", json_path)
    print("Markdown:", md_path)
    if report["aborted"]:
        print("ABORTED:", report["aborted"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
