"""
Whisper-based speech-to-text transcription module.

Provides functions to load a Whisper model and transcribe audio files
to text. Models are cached globally to avoid repeated loading.

Uses imageio-ffmpeg to decode audio files, bypassing Whisper's built-in
ffmpeg call which can fail on Windows with conda-forge's ffmpeg.
"""

import logging
import os
import subprocess

import numpy as np
import whisper

logger = logging.getLogger("voiceai.transcription")

# Global model cache: {model_size: whisper.Whisper}
_model_cache: dict = {}

# ---------------------------------------------------------------------------
# Resolve a working ffmpeg binary
# ---------------------------------------------------------------------------
_FFMPEG_EXE: str = "ffmpeg"  # fallback

try:
    import imageio_ffmpeg
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(f"Using imageio-ffmpeg binary: {_FFMPEG_EXE}")
except ImportError:
    logger.warning("imageio-ffmpeg not installed; using system ffmpeg")


def _load_audio_with_ffmpeg(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Decode any audio file to a float32 numpy array at the given sample rate.

    Uses the imageio-ffmpeg binary (or system ffmpeg) directly,
    bypassing Whisper's whisper.audio.load_audio which can crash
    on Windows with conda-forge's broken ffmpeg.
    """
    cmd = [
        _FFMPEG_EXE,
        "-nostdin",
        "-threads", "0",
        "-i", file_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-",
    ]
    logger.debug(f"ffmpeg command: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode(errors="replace") if e.stderr else "(empty)"
        logger.error(f"ffmpeg failed (exit {e.returncode}): {stderr_text}")
        raise RuntimeError(
            f"Failed to load audio with ffmpeg (exit {e.returncode}): {stderr_text}"
        ) from e

    raw_bytes = proc.stdout
    audio = np.frombuffer(raw_bytes, np.int16).flatten().astype(np.float32) / 32768.0
    logger.info(f"Decoded audio: {len(audio)} samples, {len(audio)/sr:.1f}s duration")
    return audio


def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """
    Load and cache a Whisper model.

    Args:
        model_size: Whisper model variant — one of
                    'tiny', 'base', 'small', 'medium', 'large'.

    Returns:
        A loaded whisper.Whisper model instance.
    """
    if model_size not in _model_cache:
        logger.info(f"Loading Whisper model: {model_size}")
        _model_cache[model_size] = whisper.load_model(model_size)
        logger.info(f"Whisper model '{model_size}' loaded successfully")
    return _model_cache[model_size]


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        audio_path: Path to the audio file (.wav, .mp3, .m4a, etc.).
        model_size: Whisper model variant to use.

    Returns:
        The transcribed text string, stripped of leading/trailing whitespace.
    """
    logger.info(f"--- Transcription start ---")
    logger.info(f"Audio path: {audio_path}")

    if not os.path.exists(audio_path):
        logger.error(f"Audio file does NOT exist: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    logger.info(f"File size: {file_size} bytes ({file_size / 1024:.1f} KB)")
    logger.info(f"ffmpeg binary: {_FFMPEG_EXE}")

    # Step 1: Decode audio ourselves using the working ffmpeg binary
    audio_array = _load_audio_with_ffmpeg(audio_path)

    # Step 2: Pass numpy array to Whisper (skips Whisper's own ffmpeg call)
    model = load_whisper_model(model_size)
    logger.info("Starting whisper.transcribe with pre-decoded audio array")
    result = model.transcribe(
        audio_array,
        fp16=False,       # CPU compatibility
        temperature=0,    # Deterministic decoding
    )
    text = result["text"].strip()
    logger.info(f"Transcription success: '{text[:100]}...'")
    return text
