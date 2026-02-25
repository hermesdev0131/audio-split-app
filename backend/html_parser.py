"""
HTML Parser — extracts BLOCK/BLOQUE sections from an HTML document.

Supports two formats:
1. Static HTML: headers (h1-h6) containing "BLOCK" or "BLOQUE" with sibling paragraphs.
2. JS-rendered HTML (DioApp): a `const processedBlocks = [...]` JS array in a <script> tag.
"""

import json
import re
from dataclasses import dataclass
from bs4 import BeautifulSoup

BLOCK_PATTERN = re.compile(r"\b(block|bloque)\b", re.IGNORECASE)

HEADER_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Matches: const processedBlocks = [...]
JS_BLOCKS_PATTERN = re.compile(
    r"const\s+processedBlocks\s*=\s*(\[.*?\])\s*;",
    re.DOTALL,
)


@dataclass
class Block:
    block_id: int
    header: str
    raw_text: str
    normalized_text: str

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "header": self.header,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
        }


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip, normalize punctuation."""
    text = text.lower().strip()
    # collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    # remove bracketed metadata like [music], [applause]
    text = re.sub(r"\[.*?\]", "", text)
    return text.strip()


def _extract_paragraph_text(header_tag) -> str:
    """Get the paragraph/text content that follows a block header."""
    texts: list[str] = []
    current = header_tag.find_next_sibling()

    while current:
        if current.name and current.name.lower() in HEADER_TAGS:
            break
        if current.name and current.get_text and BLOCK_PATTERN.search(current.get_text(strip=True)):
            break
        text = current.get_text(strip=True) if current.name else str(current).strip()
        if text:
            texts.append(text)
        current = current.find_next_sibling()

    if not texts:
        parent = header_tag.parent
        if parent:
            for child in parent.children:
                if child == header_tag:
                    continue
                text = child.get_text(strip=True) if hasattr(child, "get_text") else str(child).strip()
                if text:
                    texts.append(text)

    return " ".join(texts)


def _parse_js_blocks(html_content: str) -> list[Block]:
    """Extract blocks from a JavaScript `processedBlocks` array in <script> tags."""
    match = JS_BLOCKS_PATTERN.search(html_content)
    if not match:
        return []

    raw_array = match.group(1)
    try:
        items = json.loads(raw_array)
    except json.JSONDecodeError:
        return []

    blocks: list[Block] = []
    for i, text in enumerate(items):
        if not isinstance(text, str) or not text.strip():
            continue
        blocks.append(Block(
            block_id=i + 1,
            header=f"BLOQUE {i + 1}",
            raw_text=text.strip(),
            normalized_text=_normalize(text),
        ))

    return blocks


def _parse_static_html(html_content: str) -> list[Block]:
    """Extract blocks from static HTML header tags containing BLOCK/BLOQUE."""
    soup = BeautifulSoup(html_content, "html.parser")
    blocks: list[Block] = []
    block_id = 1

    headers = soup.find_all(HEADER_TAGS)

    for header in headers:
        header_text = header.get_text(strip=True)
        if not BLOCK_PATTERN.search(header_text):
            continue

        raw_text = _extract_paragraph_text(header)
        if not raw_text:
            continue

        blocks.append(Block(
            block_id=block_id,
            header=header_text,
            raw_text=raw_text,
            normalized_text=_normalize(raw_text),
        ))
        block_id += 1

    return blocks


def parse_html(html_content: str) -> list[Block]:
    """Parse HTML and extract all BLOCK/BLOQUE sections.

    Tries JS-rendered format first (processedBlocks array),
    then falls back to static HTML headers.

    Returns list of Block objects in document order.
    Raises ValueError if no blocks are found.
    """
    # Try JS format first (DioApp-style)
    blocks = _parse_js_blocks(html_content)

    # Fallback to static HTML
    if not blocks:
        blocks = _parse_static_html(html_content)

    if not blocks:
        raise ValueError("No BLOCK/BLOQUE sections found in the HTML document.")

    return blocks
