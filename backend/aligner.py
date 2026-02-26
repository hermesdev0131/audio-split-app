"""
Aligner — matches BLOCK texts to word-level timestamps from transcription.

Two-pass approach:
  Pass 1: Find high-confidence anchor blocks using token matching.
  Pass 2: Interpolate positions for low-confidence blocks between anchors.
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


def _word_sim(a: str, b: str) -> float:
    """Fast word similarity."""
    if a == b:
        return 1.0
    if abs(len(a) - len(b)) > 3:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _score_at(block_tokens: list[str], transcript_tokens: list[str], start: int) -> float:
    """Score alignment of block starting at `start`. O(n) sequential match."""
    n = len(block_tokens)
    if not n:
        return 0.0

    matched = 0.0
    t = start

    for bt in block_tokens:
        best = 0.0
        best_j = t
        for j in range(t, min(len(transcript_tokens), t + 4)):
            s = _word_sim(bt, transcript_tokens[j])
            if s > best:
                best = s
                best_j = j
            if s == 1.0:
                break
        if best >= 0.55:
            matched += best
            t = best_j + 1
        else:
            t = min(t + 1, len(transcript_tokens))

    return matched / n


def align_blocks(
    blocks: list[Block],
    transcript_words: list[WordTimestamp],
    audio_duration: float,
) -> list[AlignmentResult]:
    """Align blocks to transcript using two-pass anchor + interpolation.

    Pass 1: For each block, search broadly for the best matching position.
            Mark blocks with confidence >= 0.5 as "anchors".
    Pass 2: For non-anchor blocks between two anchors, interpolate their
            time positions proportionally based on token count.
    """
    n_blocks = len(blocks)

    if not transcript_words:
        seg = audio_duration / n_blocks
        return [
            AlignmentResult(b.block_id, b.header, i * seg, (i + 1) * seg, 0.0, 0,
                            len(_tokenize(b.normalized_text)))
            for i, b in enumerate(blocks)
        ]

    n_words = len(transcript_words)
    t_tokens = [w.word.lower().strip(".,;:!?¿¡\"'()") for w in transcript_words]

    # Pre-tokenize all blocks
    btokens = [_tokenize(b.normalized_text) for b in blocks]
    blens = [len(t) for t in btokens]
    total_btokens = sum(blens)

    # ── Pass 1: Find best position for each block ──
    raw_positions: list[tuple[int, int, float]] = []  # (start_idx, end_idx, confidence)
    min_pos = 0  # soft lower bound, not hard cursor

    for i in range(n_blocks):
        tokens = btokens[i]
        n_tok = blens[i]

        if not tokens:
            raw_positions.append((min_pos, min_pos, 0.0))
            continue

        # Proportional estimate
        prop = sum(blens[:i]) / total_btokens if total_btokens else 0
        est = int(prop * n_words)

        # Wide search: ±15% of transcript, at least ±300 words
        margin = max(300, n_words // 7)
        s_start = max(0, est - margin)
        s_end = min(n_words - 1, est + margin)

        # Also ensure we search from min_pos
        s_start = max(0, min(s_start, min_pos))

        best_score = -1.0
        best_pos = est

        # Score every 3rd position, then refine top hit
        for pos in range(s_start, s_end + 1, 3):
            sc = _score_at(tokens, t_tokens, pos)
            if sc > best_score:
                best_score = sc
                best_pos = pos
            if sc >= 0.95:
                break

        # Refine around best
        fine_s = max(s_start, best_pos - 6)
        fine_e = min(s_end, best_pos + 6)
        for pos in range(fine_s, fine_e + 1):
            sc = _score_at(tokens, t_tokens, pos)
            if sc > best_score:
                best_score = sc
                best_pos = pos

        best_pos = max(0, min(best_pos, n_words - 1))
        match_end = max(best_pos, min(best_pos + n_tok, n_words) - 1)

        raw_positions.append((best_pos, match_end, max(best_score, 0.0)))

        # Soft advance — only move forward on decent matches
        if best_score >= 0.4:
            min_pos = best_pos + 1

    # ── Pass 2: Interpolate low-confidence blocks between anchors ──
    ANCHOR_THRESHOLD = 0.45

    # Build anchor list: (block_index, start_time, end_time)
    anchors: list[tuple[int, float, float]] = []

    # Always anchor start
    anchors.append((-1, 0.0, 0.0))

    for i in range(n_blocks):
        pos_start, pos_end, conf = raw_positions[i]
        if conf >= ANCHOR_THRESHOLD:
            t_start = transcript_words[pos_start].start
            t_end = transcript_words[pos_end].end
            anchors.append((i, t_start, t_end))

    # Always anchor end
    anchors.append((n_blocks, audio_duration, audio_duration))

    # For each block, determine its time range
    results: list[AlignmentResult] = []

    for i in range(n_blocks):
        pos_start, pos_end, conf = raw_positions[i]
        n_tok = blens[i]

        if conf >= ANCHOR_THRESHOLD:
            # Use direct match
            t_start = transcript_words[pos_start].start
            t_end = transcript_words[pos_end].end
        else:
            # Interpolate between surrounding anchors
            prev_anchor = None
            next_anchor = None

            for a in anchors:
                if a[0] < i:
                    prev_anchor = a
                elif a[0] > i and next_anchor is None:
                    next_anchor = a

            if prev_anchor is None:
                prev_anchor = (-1, 0.0, 0.0)
            if next_anchor is None:
                next_anchor = (n_blocks, audio_duration, audio_duration)

            # Interpolate proportionally by token count
            blocks_between = list(range(prev_anchor[0] + 1, next_anchor[0]))
            if not blocks_between:
                blocks_between = [i]

            tokens_in_range = sum(max(1, blens[j]) for j in blocks_between)
            tokens_before_me = sum(max(1, blens[j]) for j in blocks_between if j < i)
            my_tokens = max(1, blens[i])

            time_start = prev_anchor[2]  # end of prev anchor
            time_end = next_anchor[1]    # start of next anchor
            time_span = time_end - time_start

            if tokens_in_range > 0 and time_span > 0:
                ratio_start = tokens_before_me / tokens_in_range
                ratio_end = (tokens_before_me + my_tokens) / tokens_in_range
                t_start = time_start + ratio_start * time_span
                t_end = time_start + ratio_end * time_span
            else:
                t_start = time_start
                t_end = time_start

        results.append(AlignmentResult(
            block_id=blocks[i].block_id,
            header=blocks[i].header,
            proposed_start=t_start,
            proposed_end=t_end,
            confidence=round(conf, 3),
            matched_words=int(conf * n_tok),
            total_words=n_tok,
        ))

    return results
