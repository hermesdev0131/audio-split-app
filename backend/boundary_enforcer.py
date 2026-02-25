"""
Boundary Enforcer — guarantees no overlaps, no gaps, exact duration coverage.

Rules:
  B[0]   = 0
  B[N]   = D  (total duration)
  B[i]   >= B[i-1]  (monotonic increasing)
  All boundaries are sample-accurate.
"""

from dataclasses import dataclass

from aligner import AlignmentResult


@dataclass
class SegmentBoundary:
    block_id: int
    header: str
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int
    confidence: float


def enforce_boundaries(
    alignments: list[AlignmentResult],
    audio_duration: float,
    sample_rate: int,
) -> list[SegmentBoundary]:
    """Enforce exact boundary rules on alignment results.

    Guarantees:
    - First segment starts at sample 0 (time 0)
    - Last segment ends at exact audio duration
    - No overlaps between segments
    - No gaps between segments
    - All boundaries are sample-accurate

    Args:
        alignments: Raw alignment results from the aligner.
        audio_duration: Total audio duration in seconds.
        sample_rate: Audio sample rate in Hz.

    Returns:
        List of SegmentBoundary with enforced boundaries.
    """
    n = len(alignments)
    total_samples = round(audio_duration * sample_rate)

    # Compute proposed midpoint boundaries between consecutive blocks
    # B[0] = 0, B[N] = total_samples
    # B[i] for i in 1..N-1 = midpoint between end of block i-1 and start of block i
    boundaries_samples: list[int] = [0]  # B[0]

    for i in range(1, n):
        prev_end = alignments[i - 1].proposed_end
        curr_start = alignments[i].proposed_start

        # Midpoint between previous block end and current block start
        mid_sec = (prev_end + curr_start) / 2.0
        mid_sample = round(mid_sec * sample_rate)

        # Clamp to valid range and enforce monotonic
        mid_sample = max(mid_sample, boundaries_samples[-1])
        mid_sample = min(mid_sample, total_samples)

        boundaries_samples.append(mid_sample)

    boundaries_samples.append(total_samples)  # B[N]

    # Build segments
    segments: list[SegmentBoundary] = []
    for i in range(n):
        start_sample = boundaries_samples[i]
        end_sample = boundaries_samples[i + 1]

        # Ensure minimum 1 sample per segment
        if end_sample <= start_sample and i < n - 1:
            end_sample = start_sample + 1
            # Adjust subsequent boundaries
            for j in range(i + 2, len(boundaries_samples)):
                if boundaries_samples[j] <= end_sample:
                    boundaries_samples[j] = end_sample + 1
                else:
                    break
            boundaries_samples[i + 1] = end_sample

        segments.append(SegmentBoundary(
            block_id=alignments[i].block_id,
            header=alignments[i].header,
            start_sec=start_sample / sample_rate,
            end_sec=end_sample / sample_rate,
            start_sample=start_sample,
            end_sample=end_sample,
            confidence=alignments[i].confidence,
        ))

    # Final enforcement: last segment must end at total_samples
    if segments:
        segments[-1].end_sample = total_samples
        segments[-1].end_sec = total_samples / sample_rate

    return segments


def validate_boundaries(
    segments: list[SegmentBoundary],
    audio_duration: float,
    sample_rate: int,
) -> dict:
    """Validate that all invariants hold.

    Returns a dict with boolean flags for each invariant.
    """
    total_samples = round(audio_duration * sample_rate)

    no_overlaps = True
    no_gaps = True

    for i in range(1, len(segments)):
        if segments[i].start_sample < segments[i - 1].end_sample:
            no_overlaps = False
        if segments[i].start_sample > segments[i - 1].end_sample:
            no_gaps = False

    covers_full = (
        segments[0].start_sample == 0
        and segments[-1].end_sample == total_samples
    ) if segments else False

    sum_samples = sum(s.end_sample - s.start_sample for s in segments)
    exact_match = sum_samples == total_samples

    return {
        "no_overlaps": no_overlaps,
        "no_gaps": no_gaps,
        "covers_full_audio": covers_full,
        "exact_duration_match": exact_match,
    }
