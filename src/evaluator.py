"""
Pipeline orchestrator for Voice AI evaluation.

Provides functions to evaluate individual audio samples and run
batch evaluation across an entire dataset, producing per-file
results and an aggregate summary.
"""

import json
import logging
import os
import warnings

from src.transcription import transcribe_audio
from src.llm_inference import generate_response
from src.metrics import compute_wer, measure_latency, semantic_similarity, detect_hallucination

logger = logging.getLogger("voiceai.evaluator")


def evaluate_sample(
    audio_path: str,
    ground_truth_transcript: str,
    expected_answer: str,
    whisper_model: str = "base",
    llm_model: str = "llama3.2",
) -> dict:
    """
    Evaluate a single audio sample through the full pipeline.

    Steps:
        1. Transcribe audio with Whisper.
        2. Generate LLM response from the transcription.
        3. Compute WER between ground-truth transcript and Whisper output.
        4. Compute semantic similarity between expected answer and LLM response.
        5. Detect hallucination based on similarity score.

    Args:
        audio_path: Path to the .wav audio file.
        ground_truth_transcript: Reference transcript text.
        expected_answer: Reference LLM answer text.
        whisper_model: Whisper model variant to use.
        llm_model: Ollama model name to use.

    Returns:
        A dict with keys: transcription, llm_response, latency, wer,
        semantic_similarity, hallucination.
    """
    # Step 1: Transcribe
    transcription = transcribe_audio(audio_path, model_size=whisper_model)

    # Step 2: Generate LLM response (with latency measurement)
    llm_response, latency = measure_latency(generate_response, transcription, llm_model)

    # Step 3: Compute WER
    wer = compute_wer(ground_truth_transcript, transcription)

    # Step 4: Compute semantic similarity
    sim_score = semantic_similarity(expected_answer, llm_response)

    # Step 5: Detect hallucination
    hallucination = detect_hallucination(sim_score)

    return {
        "transcription": transcription,
        "llm_response": llm_response,
        "latency": latency,
        "wer": wer,
        "semantic_similarity": sim_score,
        "hallucination": hallucination,
    }


def evaluate_batch(
    dataset: dict,
    audio_dir: str,
    whisper_model: str = "base",
    llm_model: str = "llama3.2",
) -> dict:
    """
    Evaluate all samples in a dataset.

    Iterates over the dataset entries, skips any whose audio file is
    missing (with a warning), runs evaluate_sample on each, and appends
    an aggregate '__summary__' entry.

    Args:
        dataset: Dict mapping filename → {transcript, expected_answer}.
        audio_dir: Directory containing the audio files.
        whisper_model: Whisper model variant to use.
        llm_model: Ollama model name to use.

    Returns:
        A dict of per-file results keyed by filename, plus a '__summary__'
        entry with aggregate statistics.
    """
    results: dict = {}
    total_wer = 0.0
    total_sim = 0.0
    total_latency = 0.0
    hallucination_count = 0
    processed = 0

    for filename, data in dataset.items():
        audio_path = os.path.join(audio_dir, filename)

        if not os.path.isfile(audio_path):
            warnings.warn(f"Audio file not found, skipping: {audio_path}")
            continue

        result = evaluate_sample(
            audio_path=audio_path,
            ground_truth_transcript=data["transcript"],
            expected_answer=data["expected_answer"],
            whisper_model=whisper_model,
            llm_model=llm_model,
        )
        results[filename] = result

        total_wer += result["wer"]
        total_sim += result["semantic_similarity"]
        total_latency += result["latency"]
        if result["hallucination"]:
            hallucination_count += 1
        processed += 1

    # Aggregate summary
    if processed > 0:
        results["__summary__"] = {
            "total_samples": processed,
            "avg_wer": round(total_wer / processed, 4),
            "avg_semantic_similarity": round(total_sim / processed, 4),
            "avg_latency": round(total_latency / processed, 4),
            "hallucination_count": hallucination_count,
            "hallucination_rate": round(hallucination_count / processed, 4),
        }
    else:
        results["__summary__"] = {
            "total_samples": 0,
            "avg_wer": 0.0,
            "avg_semantic_similarity": 0.0,
            "avg_latency": 0.0,
            "hallucination_count": 0,
            "hallucination_rate": 0.0,
        }

    return results
