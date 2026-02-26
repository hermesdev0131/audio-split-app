"""
Silence Detector — finds silence gaps in audio for optimal cut placement.

Analyzes RMS energy in small frames to locate low-energy regions,
then snaps proposed boundaries to the center of the nearest silence.
"""

import struct
import wave
import math


def compute_rms_frames(wav_path: str, frame_duration_ms: int = 20) -> tuple[list[float], int, int]:
    """Compute RMS energy for each short frame of a WAV file.

    Args:
        wav_path: Path to mono 16-bit WAV file.
        frame_duration_ms: Duration of each analysis frame in milliseconds.

    Returns:
        (rms_values, sample_rate, samples_per_frame)
    """
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    bytes_per_sample = sample_width * n_channels
    total_samples = len(raw) // bytes_per_sample

    # Decode all samples (16-bit signed)
    fmt = f"<{total_samples}h"
    samples = struct.unpack(fmt, raw[:total_samples * 2])

    rms_values: list[float] = []
    for start in range(0, total_samples, samples_per_frame):
        end = min(start + samples_per_frame, total_samples)
        chunk = samples[start:end]
        if not chunk:
            break
        mean_sq = sum(s * s for s in chunk) / len(chunk)
        rms_values.append(math.sqrt(mean_sq))

    return rms_values, sample_rate, samples_per_frame


def find_silence_at(
    rms_values: list[float],
    samples_per_frame: int,
    target_sample: int,
    search_radius_samples: int,
    silence_threshold_ratio: float = 0.15,
    min_silence_frames: int = 3,
) -> int | None:
    """Find the center of the best silence region near a target sample position.

    Args:
        rms_values: RMS energy per frame.
        samples_per_frame: Samples per RMS frame.
        target_sample: The proposed cut point in samples.
        search_radius_samples: How far to search in each direction (samples).
        silence_threshold_ratio: Frames below this ratio of the median RMS are "silent".
        min_silence_frames: Minimum consecutive silent frames to qualify.

    Returns:
        Sample index at the center of the best silence, or None if no silence found.
    """
    if not rms_values:
        return None

    # Compute threshold from overall median RMS
    sorted_rms = sorted(rms_values)
    median_rms = sorted_rms[len(sorted_rms) // 2] if sorted_rms else 1.0
    threshold = median_rms * silence_threshold_ratio

    # Convert search range to frame indices
    target_frame = target_sample // samples_per_frame
    radius_frames = search_radius_samples // samples_per_frame
    start_frame = max(0, target_frame - radius_frames)
    end_frame = min(len(rms_values), target_frame + radius_frames + 1)

    # Find all silence runs in the search window
    silence_runs: list[tuple[int, int]] = []  # (start_frame, end_frame)
    run_start = None

    for i in range(start_frame, end_frame):
        if rms_values[i] <= threshold:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if i - run_start >= min_silence_frames:
                    silence_runs.append((run_start, i))
                run_start = None

    # Close final run
    if run_start is not None and end_frame - run_start >= min_silence_frames:
        silence_runs.append((run_start, end_frame))

    if not silence_runs:
        # Relax: try single-frame minimum
        for i in range(start_frame, end_frame):
            if rms_values[i] <= threshold:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    silence_runs.append((run_start, i))
                    run_start = None
        if run_start is not None:
            silence_runs.append((run_start, end_frame))

    if not silence_runs:
        # Last resort: find the frame with minimum energy in the search window
        if start_frame < end_frame:
            min_frame = min(range(start_frame, end_frame), key=lambda i: rms_values[i])
            return min_frame * samples_per_frame + samples_per_frame // 2
        return None

    # Pick the silence run closest to the target
    best_run = None
    best_dist = float("inf")

    for s, e in silence_runs:
        center_frame = (s + e) // 2
        dist = abs(center_frame - target_frame)
        if dist < best_dist:
            best_dist = dist
            best_run = (s, e)

    if best_run is None:
        return None

    # Return center sample of the best silence run
    center_frame = (best_run[0] + best_run[1]) // 2
    return center_frame * samples_per_frame + samples_per_frame // 2


def snap_boundaries_to_silence(
    boundaries_samples: list[int],
    wav_path: str,
    total_samples: int,
    sample_rate: int,
    search_radius_sec: float = 1.5,
) -> list[int]:
    """Adjust boundary positions to land in silence gaps.

    Args:
        boundaries_samples: List of boundary sample positions [B0, B1, ..., BN].
                           B[0]=0 and B[N]=total_samples are kept fixed.
        wav_path: Path to the WAV file for RMS analysis.
        total_samples: Total number of samples in audio.
        sample_rate: Audio sample rate.
        search_radius_sec: How many seconds to search around each boundary.

    Returns:
        Adjusted boundary list, still monotonic, with B[0]=0 and B[N]=total_samples.
    """
    if len(boundaries_samples) <= 2:
        return boundaries_samples

    rms_values, _, samples_per_frame = compute_rms_frames(wav_path, frame_duration_ms=20)
    search_radius = int(search_radius_sec * sample_rate)

    adjusted = [boundaries_samples[0]]  # Keep B[0] = 0

    for i in range(1, len(boundaries_samples) - 1):
        target = boundaries_samples[i]
        snapped = find_silence_at(
            rms_values, samples_per_frame, target, search_radius,
        )

        if snapped is not None:
            # Enforce monotonic: must be > previous and < next original
            snapped = max(snapped, adjusted[-1] + 1)
            snapped = min(snapped, boundaries_samples[i + 1] - 1)
            adjusted.append(snapped)
        else:
            # Keep original if no silence found
            adjusted.append(max(target, adjusted[-1] + 1))

    adjusted.append(total_samples)  # Keep B[N] = total_samples
    return adjusted
