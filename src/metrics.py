"""
Evaluation metrics module.

Provides functions for computing Word Error Rate (WER), measuring
execution latency, calculating semantic similarity, and detecting
hallucinations.
"""

import time
from typing import Any, Callable, Tuple

import jiwer
from sentence_transformers import SentenceTransformer, util

# Global sentence-transformer model cache
_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers model."""
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis texts.

    Both strings are lowercased before comparison for case-insensitive
    evaluation.

    Args:
        reference: The ground-truth text.
        hypothesis: The predicted / transcribed text.

    Returns:
        WER as a float rounded to 4 decimal places.
    """
    ref = reference.lower().strip()
    hyp = hypothesis.lower().strip()
    error_rate = jiwer.wer(ref, hyp)
    return round(error_rate, 4)


def measure_latency(fn: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """
    Measure the wall-clock execution time of a callable.

    Args:
        fn: The function to execute and time.
        *args: Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        A tuple of (result, elapsed_seconds) where elapsed_seconds is
        rounded to 4 decimal places.
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, round(elapsed, 4)


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using sentence embeddings.

    Uses the 'all-MiniLM-L6-v2' model from sentence-transformers.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        A similarity score in [0, 1], rounded to 4 decimal places.
    """
    model = _get_st_model()
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity.item()), 4)


def detect_hallucination(similarity_score: float, threshold: float = 0.4) -> bool:
    """
    Determine whether an LLM response is a hallucination based on
    semantic similarity to the expected answer.

    Args:
        similarity_score: Cosine similarity between LLM response and
                          expected answer (0–1).
        threshold: Similarity below this value is flagged as hallucination.

    Returns:
        True if the response is likely a hallucination, False otherwise.
    """
    return similarity_score < threshold
