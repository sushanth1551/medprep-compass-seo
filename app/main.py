"""Command-line entry point. Run with: python -m app.main"""

from app.config import build_provider, get_provider_name, load_local_env, regenerate_enabled
from app.metrics import DEFAULT_RUNS_DIR
from app.models import ConfigError, ProviderError, ValidationError
from app.workflow import run_workflow


def main():
    print("=== MedPrep Compass — SEO Blog Draft Generator ===")
    load_local_env()
    try:
        provider = build_provider()
        regenerate = regenerate_enabled()
    except ConfigError as exc:
        print(f"\nConfiguration problem: {exc}")
        return 2
    name = get_provider_name()
    print(f"Provider selected: {name} -> {provider.name}")
    if name == "mock":
        print("Mock provider: output is demo text, not real AI content.")
    else:
        print(f"{name.capitalize()} provider: this sends a real API request (may cost money).")
    print()
    topic = input("Blog topic: ")
    keyword = input("Target keyword (press Enter to reuse the topic): ")

    try:
        result = run_workflow(
            topic, keyword, provider=provider, metrics_dir=DEFAULT_RUNS_DIR,
            regenerate_failed=regenerate,
        )
    except ValidationError as exc:
        print(f"\nInput problem: {exc}")
        return 1
    except ProviderError as exc:
        # No fallback to mock output: a failed real request produces no draft.
        print(f"\nProvider problem: {exc}")
        return 1
    except OSError as exc:
        print(f"\nFile problem: {exc}")
        return 1

    usage = result.usage
    metrics = result.metrics
    print("\nDraft saved to:", result.output_path)
    print("Structure checks passed:", "yes" if result.checks_passed else "NO")
    for check, ok in result.check_results.items():
        print(f"  [{'x' if ok else ' '}] {check}")
    if not metrics["appears_complete"]:
        print("WARNING: the draft may be incomplete: " + " ".join(metrics["incomplete_reasons"]))

    if usage.usage_source == "actual":
        print(
            f"Token usage {usage.source_label}: input={usage.actual_input_tokens}, "
            f"output={usage.actual_output_tokens}"
        )
    elif usage.usage_source == "unavailable":
        print(f"Token usage {usage.source_label}")
    else:
        print(
            f"Token usage {usage.source_label}: input={usage.estimated_input_tokens}, "
            f"output={usage.estimated_output_tokens}"
        )
    print(f"Model calls: {metrics['model_calls']}   Elapsed: {metrics['elapsed_seconds']}s")
    cost = metrics["cost"]
    amount = "unknown" if cost["amount"] is None else f"{cost['amount']:.6f} {cost['currency']}"
    print(f"Cost: {amount} — {cost['note']}")

    report = result.seo_report
    print(f"\nSEO checklist: {report.score_text}")
    print(f"Approx. word count: {report.word_count}")
    for label, items in (("FAILED", report.failed), ("WARNING", report.warnings)):
        for item in items:
            print(f"  [{label}] {item}")
    for item in report.passed:
        print(f"  [passed] {item}")
    print(f"Fact/source verification: {report.source_verification}")
    for sentence in report.claims_needing_evidence[:5]:
        print(f"  [needs evidence] {sentence[:120]}")
    print("SEO report saved to:", result.report_path)
    print("Run metrics saved to:", result.metrics_path)
    print("Nothing was published. Review the draft before using it anywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
