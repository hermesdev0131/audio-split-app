"""
Transcriber — uses faster-whisper (CTranslate2) for ~4x faster transcription
with word-level timestamps.
"""

import subprocess
import os
from dataclasses import dataclass

from faster_whisper import WhisperModel


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


# Cache model in memory so it's loaded only once across jobs
_model_cache: dict[str, WhisperModel] = {}


def _get_model(model_name: str) -> WhisperModel:
    if model_name not in _model_cache:
        _model_cache[model_name] = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",  # fastest on CPU
        )
    return _model_cache[model_name]


def get_audio_duration(audio_path: str) -> float:
    """Get exact audio duration in seconds using FFprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def get_sample_rate(audio_path: str) -> int:
    """Get audio sample rate using FFprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")
    return int(result.stdout.strip())


def transcribe(audio_path: str, model_name: str = "base") -> list[WordTimestamp]:
    """Transcribe audio and return word-level timestamps.

    Uses faster-whisper (CTranslate2) for significantly faster inference.

    Args:
        audio_path: Path to audio file (any supported format).
        model_name: Whisper model size (tiny, base, small, medium, large-v3).

    Returns:
        List of WordTimestamp objects sorted by start time.
    """
    model = _get_model(model_name)

    segments_iter, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,          # skip silence regions (faster)
        vad_parameters=dict(
            min_silence_duration_ms=300,
        ),
        beam_size=5,              # good accuracy without being slow
        language="es",            # skip language detection
    )

    words: list[WordTimestamp] = []
    for segment in segments_iter:
        if segment.words:
            for w in segment.words:
                words.append(WordTimestamp(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                ))

    return words
