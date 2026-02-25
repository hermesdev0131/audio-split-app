"""
Audio Cutter — sample-accurate slicing using FFmpeg.

Converts to WAV PCM first, then slices by sample indices.
Optionally encodes output segments to MP3.
"""

import os
import subprocess
import struct
import wave
from pathlib import Path

from boundary_enforcer import SegmentBoundary


def convert_to_wav_pcm(input_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """Convert any audio to WAV PCM mono at the given sample rate."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        "-f", "wav",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    return output_path


def cut_segments(
    audio_path: str,
    segments: list[SegmentBoundary],
    output_dir: str,
    output_format: str = "wav",
) -> list[str]:
    """Cut audio into segments using sample-accurate boundaries.

    Uses raw PCM data for exact sample-level slicing, then writes
    proper WAV files. Optionally converts to MP3.

    Args:
        audio_path: Path to the source audio file.
        segments: List of SegmentBoundary with sample-accurate boundaries.
        output_dir: Directory to write output segment files.
        output_format: "wav" or "mp3".

    Returns:
        List of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the source WAV
    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        all_frames = wf.readframes(n_frames)

    bytes_per_sample = sample_width * n_channels
    output_files: list[str] = []

    for seg in segments:
        start_byte = seg.start_sample * bytes_per_sample
        end_byte = seg.end_sample * bytes_per_sample

        # Clamp to valid range
        start_byte = max(0, start_byte)
        end_byte = min(len(all_frames), end_byte)

        segment_frames = all_frames[start_byte:end_byte]
        n_segment_frames = (end_byte - start_byte) // bytes_per_sample

        # Write WAV segment
        filename = f"{seg.block_id:03d}.wav"
        wav_path = os.path.join(output_dir, filename)

        with wave.open(wav_path, "wb") as out_wf:
            out_wf.setnchannels(n_channels)
            out_wf.setsampwidth(sample_width)
            out_wf.setframerate(frame_rate)
            out_wf.writeframes(segment_frames)

        if output_format == "mp3":
            mp3_path = os.path.join(output_dir, f"{seg.block_id:03d}.mp3")
            cmd = [
                "ffmpeg", "-y",
                "-i", wav_path,
                "-codec:a", "libmp3lame",
                "-q:a", "2",
                mp3_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"MP3 encoding failed: {result.stderr}")
            os.remove(wav_path)
            output_files.append(mp3_path)
        else:
            output_files.append(wav_path)

    return output_files
