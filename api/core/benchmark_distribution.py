"""WP14 §14.3 — Compare generated benchmark stats to target distribution; emit warnings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.benchmark.generator import BenchmarkStats

# Target shares (§14.3)
TARGET_DIFFICULTY: dict[str, float] = {"L1": 0.40, "L2": 0.40, "L3": 0.20}
TARGET_DOC_TYPE: dict[str, float] = {
    "structured_manual": 0.25,
    "operational_guide": 0.20,
    "module_manual": 0.20,
    "changelog_as_manual": 0.15,
}
# Remainder (~20%) = "others" bucket (any other doc_type)
OTHERS_DOC_SHARE = 1.0 - sum(TARGET_DOC_TYPE.values())
MIN_VALIDATED_PRODUCTION = 300


def benchmark_distribution_report(stats: "BenchmarkStats") -> dict:
    """Structured report + human-readable warnings for API / UI."""
    n = stats.total_pairs
    warnings: list[str] = []
    difficulty_pct: dict[str, float] = {}
    doc_type_pct: dict[str, float] = {}

    if n == 0:
        return {
            "warnings": [],
            "difficulty_actual_pct": {},
            "doc_type_actual_pct": {},
            "validated_count": 0,
            "note": "No pairs generated.",
        }

    for label in ("L1", "L2", "L3"):
        c = stats.by_difficulty.get(label, 0)
        difficulty_pct[label] = round(100.0 * c / n, 1)

    for dt, c in stats.by_doc_type.items():
        doc_type_pct[dt] = round(100.0 * c / n, 1)

    # Slack scales with sample size (small runs are noisy)
    slack = max(0.08, min(0.20, 24.0 / max(n, 1)))

    if n >= 20:
        for label, target in TARGET_DIFFICULTY.items():
            actual = stats.by_difficulty.get(label, 0) / n
            if abs(actual - target) > slack:
                warnings.append(
                    f"Difficulty {label}: {actual*100:.1f}% vs target ~{target*100:.0f}% "
                    f"(n={n}, slack ±{slack*100:.0f}pp)"
                )

    if n >= 20:
        typed_keys = set(TARGET_DOC_TYPE)
        typed_count = sum(stats.by_doc_type.get(dt, 0) for dt in typed_keys)
        others_count = n - typed_count
        for dt, target in TARGET_DOC_TYPE.items():
            actual = stats.by_doc_type.get(dt, 0) / n
            if abs(actual - target) > slack:
                warnings.append(
                    f"Doc type {dt}: {actual*100:.1f}% vs target ~{target*100:.0f}%"
                )
        actual_others = others_count / n
        if abs(actual_others - OTHERS_DOC_SHARE) > slack + 0.05:
            warnings.append(
                f"Other doc types: {actual_others*100:.1f}% vs target ~{OTHERS_DOC_SHARE*100:.0f}%"
            )

    if n >= 80 and stats.validated_count < MIN_VALIDATED_PRODUCTION:
        warnings.append(
            f"Validated pairs: {stats.validated_count} — spec recommends ≥{MIN_VALIDATED_PRODUCTION} "
            "for a production benchmark (increase max_pairs or run multiple batches)."
        )

    if n < 80:
        warnings.append(
            f"Sample size n={n}: distribution vs §14.3 targets is indicative only; run ≥80+ pairs for stable mix."
        )

    return {
        "warnings": warnings,
        "difficulty_actual_pct": difficulty_pct,
        "doc_type_actual_pct": doc_type_pct,
        "difficulty_target_pct": {k: round(v * 100, 0) for k, v in TARGET_DIFFICULTY.items()},
        "doc_type_target_pct": {k: round(v * 100, 0) for k, v in TARGET_DOC_TYPE.items()},
        "others_doc_target_pct": round(OTHERS_DOC_SHARE * 100, 0),
        "validated_count": stats.validated_count,
        "min_validated_production": MIN_VALIDATED_PRODUCTION,
    }
