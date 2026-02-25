"""
Transcriber — uses OpenAI Whisper to produce word-level timestamps.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path

import whisper


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


def _convert_to_wav_16k(input_path: str, output_path: str) -> None:
    """Convert any supported audio to WAV mono 16kHz using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")


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

    Args:
        audio_path: Path to audio file (any supported format).
        model_name: Whisper model size (tiny, base, small, medium, large).

    Returns:
        List of WordTimestamp objects sorted by start time.
    """
    # Convert to WAV 16kHz for Whisper
    tmp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, "audio.wav")

    try:
        _convert_to_wav_16k(audio_path, wav_path)

        model = whisper.load_model(model_name)
        result = model.transcribe(
            wav_path,
            word_timestamps=True,
            verbose=False,
        )

        words: list[WordTimestamp] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append(WordTimestamp(
                    word=w["word"].strip(),
                    start=w["start"],
                    end=w["end"],
                ))

        return words

    finally:
        # Clean up temp file
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
