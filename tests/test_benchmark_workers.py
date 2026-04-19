"""Benchmark parallel retrieval (workers / concurrency)."""

from api.benchmark.evaluator import resolve_benchmark_workers


def test_resolve_benchmark_workers_explicit_clamped():
    assert resolve_benchmark_workers(1) == 1
    assert resolve_benchmark_workers(32) == 32
    assert resolve_benchmark_workers(100) == 32
