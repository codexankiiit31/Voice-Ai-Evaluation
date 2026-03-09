"""
FastAPI backend for the Voice AI Evaluation Pipeline.

Exposes REST endpoints for single/batch evaluation, report retrieval,
and Ollama model listing.
"""

import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import traceback
from typing import Optional

import numpy as np
import requests as http_requests
from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.evaluator import evaluate_sample, evaluate_batch

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voiceai")

# ---------------------------------------------------------------------------
# Deterministic seeds
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Voice AI Evaluation API",
    description="REST API for evaluating Voice AI pipelines with Whisper + LangChain/Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    audio_dir: str = "dataset/audio"
    ground_truth_path: str = "dataset/ground_truth.json"
    whisper_model: str = "base"
    llm_model: str = "llama3.2"


# ---------------------------------------------------------------------------
# Helper: save batch results to disk
# ---------------------------------------------------------------------------
REPORTS_DIR = "reports"


def _save_report(results: dict) -> None:
    """Persist evaluation results to reports/evaluation_results.json."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Dataset Auto-Save Helper
# ---------------------------------------------------------------------------
def _save_to_dataset(audio_file_path: str, original_filename: str, transcript: str, expected_answer: str):
    """
    Saves an evaluated audio file into the dataset/audio folder and 
    appends its ground truth data to dataset/ground_truth.json.
    """
    audio_dir = "dataset/audio"
    gt_path = "dataset/ground_truth.json"
    
    os.makedirs(audio_dir, exist_ok=True)
    safe_filename = original_filename or f"audio_{os.getpid()}.wav"
    dest_audio_path = os.path.join(audio_dir, safe_filename)
    
    try:
        shutil.copy2(audio_file_path, dest_audio_path)
        logger.info(f"Saved audio to dataset: {dest_audio_path}")
    except Exception as e:
        logger.error(f"Failed to copy audio to dataset: {e}")
        
    gt_data = {}
    if os.path.exists(gt_path):
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except json.JSONDecodeError:
            pass
            
    gt_data[safe_filename] = {
        "transcript": transcript,
        "expected_answer": expected_answer
    }
    
    try:
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=4)
        logger.info(f"Updated ground truth in: {gt_path}")
    except Exception as e:
        logger.error(f"Failed to update ground truth: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Health-check endpoint."""
    return {"status": "ok", "message": "Voice AI Evaluation API is running"}


@app.post("/evaluate/single")
async def evaluate_single(
    audio_file: UploadFile = File(...),
    ground_truth_transcript: str = Form(...),
    expected_answer: str = Form(...),
    whisper_model: str = Form("base"),
    llm_model: str = Form("llama3.2"),
):
    """
    Evaluate a single uploaded audio file.

    Returns transcription, LLM response, and all evaluation metrics.
    """
    # Save uploaded file to a temp location
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"voiceai_{os.getpid()}_{id(audio_file)}{suffix}")
    logger.info("=== /evaluate/single ===")
    logger.info(f"Filename: {audio_file.filename}, Content-Type: {audio_file.content_type}")
    logger.info(f"Suffix: {suffix}, Temp path: {tmp_path}")
    try:
        content = await audio_file.read()
        logger.info(f"Read {len(content)} bytes from upload")
        with open(tmp_path, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Verify the file was written correctly
        written_size = os.path.getsize(tmp_path)
        logger.info(f"Wrote {written_size} bytes to {tmp_path}")
        logger.info(f"File exists: {os.path.exists(tmp_path)}, Size matches: {written_size == len(content)}")

        # Quick ffmpeg probe to check if the file is valid
        try:
            probe = subprocess.run(
                ["ffmpeg", "-i", tmp_path, "-f", "null", "-"],
                capture_output=True, text=True, timeout=10,
            )
            logger.debug(f"ffmpeg probe stderr: {probe.stderr[:500]}")
        except Exception as probe_err:
            logger.warning(f"ffmpeg probe failed: {probe_err}")

        result = evaluate_sample(
            audio_path=tmp_path,
            ground_truth_transcript=ground_truth_transcript,
            expected_answer=expected_answer,
            whisper_model=whisper_model,
            llm_model=llm_model,
        )
        
        # Save to dataset for future batch evaluations
        if audio_file.filename:
            _save_to_dataset(
                audio_file_path=tmp_path,
                original_filename=audio_file.filename,
                transcript=ground_truth_transcript,
                expected_answer=expected_answer
            )
            
        result["filename"] = audio_file.filename
        logger.info("Evaluation complete successfully")
        return result

    except Exception as e:
        logger.error(f"Evaluation FAILED: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/evaluate/speak")
async def evaluate_speak(
    audio_file: UploadFile = File(...),
    ground_truth_transcript: Optional[str] = Form(None),
    expected_answer: Optional[str] = Form(None),
    whisper_model: str = Form("base"),
    llm_model: str = Form("llama3.2"),
):
    """
    Evaluate a mic-recorded audio file.

    Ground truth and expected answer are optional.
    - If provided: returns full evaluation (transcription, LLM response, metrics).
    - If not provided: returns only transcription, LLM response, and latency.
    """
    suffix = os.path.splitext(audio_file.filename or "recording.wav")[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"voiceai_speak_{os.getpid()}_{id(audio_file)}{suffix}")
    try:
        content = await audio_file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        if ground_truth_transcript and expected_answer:
            # Full evaluation with metrics
            result = evaluate_sample(
                audio_path=tmp_path,
                ground_truth_transcript=ground_truth_transcript,
                expected_answer=expected_answer,
                whisper_model=whisper_model,
                llm_model=llm_model,
            )
            
            # Save to dataset for future batch evaluations
            _save_to_dataset(
                audio_file_path=tmp_path,
                original_filename=audio_file.filename or f"speak_{os.getpid()}.wav",
                transcript=ground_truth_transcript,
                expected_answer=expected_answer
            )
        else:
            # Transcription + LLM response only (no ground truth)
            from src.transcription import transcribe_audio
            from src.llm_inference import generate_response
            from src.metrics import measure_latency

            transcription = transcribe_audio(tmp_path, model_size=whisper_model)
            llm_response, latency = measure_latency(
                generate_response, transcription, llm_model
            )
            result = {
                "transcription": transcription,
                "llm_response": llm_response,
                "latency": latency,
            }

        result["filename"] = audio_file.filename or "mic_recording.wav"
        result["mode"] = "full_evaluation" if (ground_truth_transcript and expected_answer) else "speak_only"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/evaluate/batch")
async def evaluate_batch_endpoint(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
):
    """
    Run batch evaluation on the full dataset.

    Reads ground-truth JSON, evaluates all audio files found in audio_dir,
    saves the report, and returns the results.
    """
    if not os.path.isfile(request.ground_truth_path):
        raise HTTPException(
            status_code=404,
            detail=f"Ground truth file not found: {request.ground_truth_path}",
        )

    with open(request.ground_truth_path, "r") as f:
        dataset = json.load(f)

    try:
        results = evaluate_batch(
            dataset=dataset,
            audio_dir=request.audio_dir,
            whisper_model=request.whisper_model,
            llm_model=request.llm_model,
        )
        # Save report in background
        background_tasks.add_task(_save_report, results)
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report")
def get_report():
    """Return the latest saved evaluation report."""
    report_path = os.path.join(REPORTS_DIR, "evaluation_results.json")
    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="No evaluation report found.")

    with open(report_path, "r") as f:
        report = json.load(f)
    return report


@app.get("/models")
def list_models():
    """
    Query the local Ollama server for available models.

    Returns a list of model names.
    """
    try:
        resp = http_requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        model_names = [m["name"] for m in data.get("models", [])]
        return {"models": model_names}
    except http_requests.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not reachable at http://localhost:11434",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
