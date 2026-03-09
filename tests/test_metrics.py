"""
Unit tests for src.metrics module.

Tests cover: compute_wer, semantic_similarity, detect_hallucination,
and measure_latency.
"""

import pytest

from src.metrics import compute_wer, semantic_similarity, detect_hallucination, measure_latency


# =====================================================================
# compute_wer
# =====================================================================

class TestComputeWer:
    def test_perfect_match(self):
        assert compute_wer("Hello world", "Hello world") == 0.0

    def test_case_insensitive(self):
        assert compute_wer("Hello World", "hello world") == 0.0

    def test_complete_mismatch(self):
        wer = compute_wer("Hello world", "foo bar baz qux")
        assert wer > 0.5

    def test_partial_mismatch(self):
        wer = compute_wer("the cat sat on the mat", "the cat sat on a mat")
        assert 0 < wer < 1.0

    def test_empty_hypothesis(self):
        wer = compute_wer("hello world", "")
        assert wer == 1.0


# =====================================================================
# semantic_similarity
# =====================================================================

class TestSemanticSimilarity:
    def test_identical_texts(self):
        score = semantic_similarity("Hello world", "Hello world")
        assert score >= 0.99

    def test_related_texts(self):
        score = semantic_similarity(
            "Paris is the capital of France.",
            "The capital city of France is Paris.",
        )
        assert score > 0.8

    def test_unrelated_texts(self):
        score = semantic_similarity(
            "The weather is sunny today.",
            "Quantum computing uses qubits.",
        )
        assert score < 0.5

    def test_range(self):
        score = semantic_similarity("Hello", "World")
        assert 0 <= score <= 1


# =====================================================================
# detect_hallucination
# =====================================================================

class TestDetectHallucination:
    def test_below_threshold(self):
        assert detect_hallucination(0.2) is True

    def test_above_threshold(self):
        assert detect_hallucination(0.8) is False

    def test_at_boundary(self):
        # At exactly the threshold → not a hallucination
        assert detect_hallucination(0.4) is False

    def test_custom_threshold(self):
        assert detect_hallucination(0.6, threshold=0.7) is True
        assert detect_hallucination(0.8, threshold=0.7) is False


# =====================================================================
# measure_latency
# =====================================================================

class TestMeasureLatency:
    def test_returns_correct_result(self):
        result, elapsed = measure_latency(lambda x: x * 2, 5)
        assert result == 10

    def test_non_negative_time(self):
        _, elapsed = measure_latency(sum, [1, 2, 3])
        assert elapsed >= 0

    def test_with_kwargs(self):
        def adder(a, b=0):
            return a + b
        result, elapsed = measure_latency(adder, 3, b=7)
        assert result == 10
        assert elapsed >= 0
