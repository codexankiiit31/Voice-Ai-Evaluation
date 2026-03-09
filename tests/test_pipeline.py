"""
Integration tests for src.evaluator module.

Uses unittest.mock.patch to mock transcribe_audio and generate_response
so that tests do not require Whisper or Ollama to be running.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.evaluator import evaluate_sample, evaluate_batch


# =====================================================================
# Fixtures & helpers
# =====================================================================

MOCK_TRANSCRIPT = "What is the capital of France?"
MOCK_LLM_RESPONSE = "Paris is the capital of France."


def _patch_transcribe(return_value: str = MOCK_TRANSCRIPT):
    return patch("src.evaluator.transcribe_audio", return_value=return_value)


def _patch_generate(return_value: str = MOCK_LLM_RESPONSE):
    return patch("src.evaluator.generate_response", return_value=return_value)


# =====================================================================
# evaluate_sample
# =====================================================================

class TestEvaluateSample:
    def test_returns_all_required_keys(self):
        with _patch_transcribe(), _patch_generate():
            result = evaluate_sample(
                audio_path="dummy.wav",
                ground_truth_transcript=MOCK_TRANSCRIPT,
                expected_answer=MOCK_LLM_RESPONSE,
            )
        required = {"transcription", "llm_response", "latency", "wer",
                     "semantic_similarity", "hallucination"}
        assert required.issubset(result.keys())

    def test_wer_zero_when_transcript_matches(self):
        with _patch_transcribe(MOCK_TRANSCRIPT), _patch_generate():
            result = evaluate_sample(
                audio_path="dummy.wav",
                ground_truth_transcript=MOCK_TRANSCRIPT,
                expected_answer=MOCK_LLM_RESPONSE,
            )
        assert result["wer"] == 0.0

    def test_no_hallucination_when_response_matches(self):
        with _patch_transcribe(), _patch_generate(MOCK_LLM_RESPONSE):
            result = evaluate_sample(
                audio_path="dummy.wav",
                ground_truth_transcript=MOCK_TRANSCRIPT,
                expected_answer=MOCK_LLM_RESPONSE,
            )
        assert result["hallucination"] is False

    def test_latency_non_negative(self):
        with _patch_transcribe(), _patch_generate():
            result = evaluate_sample(
                audio_path="dummy.wav",
                ground_truth_transcript=MOCK_TRANSCRIPT,
                expected_answer=MOCK_LLM_RESPONSE,
            )
        assert result["latency"] >= 0


# =====================================================================
# evaluate_batch
# =====================================================================

class TestEvaluateBatch:
    def test_skips_missing_audio(self, tmp_path):
        dataset = {
            "missing.wav": {
                "transcript": MOCK_TRANSCRIPT,
                "expected_answer": MOCK_LLM_RESPONSE,
            }
        }
        with _patch_transcribe(), _patch_generate():
            results = evaluate_batch(dataset, str(tmp_path))

        # missing.wav should not be in results
        assert "missing.wav" not in results
        assert results["__summary__"]["total_samples"] == 0

    def test_includes_summary(self, tmp_path):
        # Create a dummy audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        dataset = {
            "test.wav": {
                "transcript": MOCK_TRANSCRIPT,
                "expected_answer": MOCK_LLM_RESPONSE,
            }
        }
        with _patch_transcribe(), _patch_generate():
            results = evaluate_batch(dataset, str(tmp_path))

        assert "__summary__" in results

    def test_summary_has_required_keys(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        dataset = {
            "test.wav": {
                "transcript": MOCK_TRANSCRIPT,
                "expected_answer": MOCK_LLM_RESPONSE,
            }
        }
        with _patch_transcribe(), _patch_generate():
            results = evaluate_batch(dataset, str(tmp_path))

        summary = results["__summary__"]
        required_keys = {
            "total_samples", "avg_wer", "avg_semantic_similarity",
            "avg_latency", "hallucination_count", "hallucination_rate",
        }
        assert required_keys.issubset(summary.keys())

    def test_batch_processes_existing_files(self, tmp_path):
        audio1 = tmp_path / "a.wav"
        audio2 = tmp_path / "b.wav"
        audio1.write_bytes(b"\x00" * 100)
        audio2.write_bytes(b"\x00" * 100)

        dataset = {
            "a.wav": {"transcript": "hello", "expected_answer": "hi"},
            "b.wav": {"transcript": "world", "expected_answer": "earth"},
        }
        with _patch_transcribe("hello"), _patch_generate("hi"):
            results = evaluate_batch(dataset, str(tmp_path))

        assert "a.wav" in results
        assert "b.wav" in results
        assert results["__summary__"]["total_samples"] == 2
