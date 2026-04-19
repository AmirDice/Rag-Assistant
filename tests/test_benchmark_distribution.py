"""§14.3 distribution warnings."""

from __future__ import annotations

from api.benchmark.generator import BenchmarkStats
from api.core.benchmark_distribution import benchmark_distribution_report


def test_report_empty():
    r = benchmark_distribution_report(BenchmarkStats())
    assert r["warnings"] == []


def test_skewed_difficulty_warns():
    stats = BenchmarkStats(
        total_pairs=100,
        by_difficulty={"L1": 100, "L2": 0, "L3": 0},
        by_doc_type={"structured_manual": 100},
        validated_count=100,
    )
    r = benchmark_distribution_report(stats)
    assert any("L2" in w or "L3" in w or "L1" in w for w in r["warnings"])
