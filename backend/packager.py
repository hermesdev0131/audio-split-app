"""
Packager — creates ZIP archive with audio segments, cuts.json, and report.json.
"""

import json
import os
import zipfile
from pathlib import Path

from boundary_enforcer import SegmentBoundary


def create_cuts_json(
    segments: list[SegmentBoundary],
    audio_duration: float,
    invariants: dict,
) -> dict:
    """Build the cuts.json structure."""
    return {
        "audio_duration_sec": round(audio_duration, 6),
        "blocks": [
            {
                "block_id": seg.block_id,
                "header": seg.header,
                "start_sec": round(seg.start_sec, 6),
                "end_sec": round(seg.end_sec, 6),
                "start_sample": seg.start_sample,
                "end_sample": seg.end_sample,
                "confidence": seg.confidence,
            }
            for seg in segments
        ],
        "invariants": invariants,
    }


def create_report_json(
    segments: list[SegmentBoundary],
    audio_duration: float,
    whisper_model: str,
) -> dict:
    """Build an optional report.json with processing summary."""
    low_confidence = [s for s in segments if s.confidence < 0.50]
    warnings = [s for s in segments if 0.50 <= s.confidence < 0.75]

    return {
        "total_blocks": len(segments),
        "audio_duration_sec": round(audio_duration, 6),
        "whisper_model": whisper_model,
        "low_confidence_blocks": [
            {"block_id": s.block_id, "header": s.header, "confidence": s.confidence}
            for s in low_confidence
        ],
        "warning_blocks": [
            {"block_id": s.block_id, "header": s.header, "confidence": s.confidence}
            for s in warnings
        ],
        "all_ok": len(low_confidence) == 0 and len(warnings) == 0,
    }


def package_zip(
    segment_files: list[str],
    segments: list[SegmentBoundary],
    audio_duration: float,
    invariants: dict,
    whisper_model: str,
    output_path: str,
) -> str:
    """Create a ZIP file containing all segments and metadata.

    Args:
        segment_files: List of paths to audio segment files.
        segments: Segment boundary data.
        audio_duration: Total audio duration.
        invariants: Validation results dict.
        whisper_model: Name of Whisper model used.
        output_path: Path for the output ZIP file.

    Returns:
        Path to the created ZIP file.
    """
    cuts = create_cuts_json(segments, audio_duration, invariants)
    report = create_report_json(segments, audio_duration, whisper_model)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add audio segments
        for filepath in segment_files:
            zf.write(filepath, os.path.basename(filepath))

        # Add metadata
        zf.writestr("cuts.json", json.dumps(cuts, indent=2, ensure_ascii=False))
        zf.writestr("report.json", json.dumps(report, indent=2, ensure_ascii=False))

    return output_path
