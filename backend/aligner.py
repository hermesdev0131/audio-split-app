"""
Aligner — matches BLOCK texts to word-level timestamps from transcription.

Uses sliding-window token matching with fuzzy fallback.
"""

import re
from dataclasses import dataclass

from difflib import SequenceMatcher

from html_parser import Block
from transcriber import WordTimestamp


@dataclass
class AlignmentResult:
    block_id: int
    header: str
    proposed_start: float
    proposed_end: float
    confidence: float
    matched_words: int
    total_words: int


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r"[a-záéíóúñüà-ÿ']+", text.lower())


def _word_similarity(a: str, b: str) -> float:
    """Compare two words using SequenceMatcher ratio."""
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _find_best_window(
    block_tokens: list[str],
    transcript_words: list[WordTimestamp],
) -> tuple[int, int, float]:
    """Find the best matching window in the transcript for a block's tokens.

    Returns (start_index, end_index, confidence) into transcript_words.
    """
    if not block_tokens or not transcript_words:
        return 0, 0, 0.0

    n_block = len(block_tokens)
    n_transcript = len(transcript_words)
    transcript_tokens = [w.word.lower().strip(".,;:!?¿¡\"'()") for w in transcript_words]

    best_score = -1.0
    best_start = 0
    best_end = 0

    # Sliding window: try windows of size n_block ± 30%
    min_window = max(1, int(n_block * 0.7))
    max_window = min(n_transcript, int(n_block * 1.3) + 1)

    for window_size in range(min_window, max_window + 1):
        for start in range(0, n_transcript - window_size + 1):
            window_tokens = transcript_tokens[start : start + window_size]

            # Score: average similarity of best-matched pairs
            score = _score_alignment(block_tokens, window_tokens)

            if score > best_score:
                best_score = score
                best_start = start
                best_end = start + window_size - 1

    return best_start, best_end, best_score


def _score_alignment(block_tokens: list[str], window_tokens: list[str]) -> float:
    """Score how well block_tokens match window_tokens using sequential matching."""
    if not block_tokens or not window_tokens:
        return 0.0

    matched = 0
    w_idx = 0

    for b_token in block_tokens:
        best_sim = 0.0
        best_pos = -1

        # Search forward in window from current position
        search_end = min(len(window_tokens), w_idx + 5)
        for j in range(w_idx, search_end):
            sim = _word_similarity(b_token, window_tokens[j])
            if sim > best_sim:
                best_sim = sim
                best_pos = j

        if best_sim >= 0.6:
            matched += best_sim
            if best_pos >= 0:
                w_idx = best_pos + 1
        # If no match found, skip this block token (allow gaps)

    return matched / len(block_tokens) if block_tokens else 0.0


def align_blocks(
    blocks: list[Block],
    transcript_words: list[WordTimestamp],
    audio_duration: float,
) -> list[AlignmentResult]:
    """Align each block to a time range in the transcript.

    Args:
        blocks: Parsed BLOCK sections from HTML.
        transcript_words: Word-level timestamps from Whisper.
        audio_duration: Total audio duration in seconds.

    Returns:
        List of AlignmentResult, one per block, in order.
    """
    if not transcript_words:
        # Fallback: equal segmentation
        segment_duration = audio_duration / len(blocks)
        return [
            AlignmentResult(
                block_id=b.block_id,
                header=b.header,
                proposed_start=i * segment_duration,
                proposed_end=(i + 1) * segment_duration,
                confidence=0.0,
                matched_words=0,
                total_words=len(_tokenize(b.normalized_text)),
            )
            for i, b in enumerate(blocks)
        ]

    results: list[AlignmentResult] = []
    # Track which portion of the transcript we've consumed to enforce ordering
    search_start = 0

    for block in blocks:
        block_tokens = _tokenize(block.normalized_text)

        if not block_tokens:
            results.append(AlignmentResult(
                block_id=block.block_id,
                header=block.header,
                proposed_start=0.0,
                proposed_end=0.0,
                confidence=0.0,
                matched_words=0,
                total_words=0,
            ))
            continue

        # Search only in the remaining transcript portion
        remaining = transcript_words[search_start:]
        start_idx, end_idx, confidence = _find_best_window(block_tokens, remaining)

        # Convert to absolute indices
        abs_start = search_start + start_idx
        abs_end = search_start + end_idx

        proposed_start = transcript_words[abs_start].start
        proposed_end = transcript_words[abs_end].end

        results.append(AlignmentResult(
            block_id=block.block_id,
            header=block.header,
            proposed_start=proposed_start,
            proposed_end=proposed_end,
            confidence=round(confidence, 3),
            matched_words=int(confidence * len(block_tokens)),
            total_words=len(block_tokens),
        ))

        # Move search start forward (allow some overlap for safety)
        search_start = max(search_start, abs_end - 5)

    return results
