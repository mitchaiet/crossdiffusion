"""Fast local clue transcription with Tesseract and grid numbering checks."""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .grid import Puzzle
from .gridvision import GridDetection

_HEADING = re.compile(r"^\s*(ACROSS|DOWN)\s*$", re.IGNORECASE)
_CLUE = re.compile(r"^\s*(\d{1,3})\s*[.,):]\s*(\S.*)$")
_PAGE = re.compile(r"^\[Page \d+\]$", re.IGNORECASE)


def _command() -> str | None:
    configured = os.environ.get("CROSSWORD_TESSERACT")
    if configured:
        return configured if Path(configured).is_file() else None
    found = shutil.which("tesseract")
    if found:
        return found
    local = Path.home() / ".local/bin/tesseract"
    return str(local) if local.is_file() else None


def available() -> bool:
    return _command() is not None


def _parse(text: str) -> dict[str, list[tuple[int, str]]]:
    sections: dict[str, list[tuple[int, str]]] = {"across": [], "down": []}
    direction = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _PAGE.fullmatch(line):
            direction = None
            continue
        heading = _HEADING.fullmatch(line)
        if heading:
            direction = heading.group(1).lower()
            continue
        if direction is None:
            continue
        match = _CLUE.match(line)
        if match:
            sections[direction].append((int(match.group(1)), match.group(2).strip()))
        elif sections[direction]:
            number, previous = sections[direction][-1]
            sections[direction][-1] = (number, previous + " " + line)
    return sections


def _number_distance(read: int, expected: int) -> int:
    a, b = str(read), str(expected)
    return sum(x != y for x, y in zip(a, b, strict=True)) if len(a) == len(b) else 9


def _align(sections: dict[str, list[tuple[int, str]]], puzzle: Puzzle) -> dict | None:
    """Fix isolated OCR digit mistakes only when the full clue sequence is present."""
    clues: dict = {"across": {}, "down": {}}
    for direction, code in (("across", "A"), ("down", "D")):
        expected = sorted(slot.number for slot in puzzle.slots if slot.direction == code)
        read = sections[direction]
        if len(read) != len(expected):
            return None
        exact = sum(number == wanted for (number, _), wanted in zip(read, expected, strict=True))
        if exact < max(1, math.ceil(len(expected) * 0.8)):
            return None
        if any(_number_distance(number, wanted) > 1 for (number, _), wanted in zip(read, expected, strict=True)):
            return None
        for (_, text), wanted in zip(read, expected, strict=True):
            if not text:
                return None
            clues[direction][str(wanted)] = text
    return clues


def _recognize(image: Image.Image, psm: int, upscale: bool = False) -> str:
    command = _command()
    if command is None:
        raise RuntimeError("Tesseract is not installed")
    if upscale:
        factor = min(2.0, 3000 / max(image.size))
        if factor > 1.0:
            image = image.resize((int(image.width * factor), int(image.height * factor)),
                                 Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = subprocess.run(
        [command, "stdin", "stdout", "-l", "eng", "--psm", str(psm)],
        input=buffer.getvalue(), capture_output=True, timeout=20, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[:200])
    return result.stdout.decode("utf-8", errors="replace")


def _side_by_side(image: Image.Image, detection: GridDetection) -> list[Image.Image]:
    """Crop the clue columns to the right of the detected grid."""
    factor = 1400 / max(image.size)
    factor = min(factor, 2.5) if factor < 1 or factor > 1.2 else 1.0
    right = int(detection.bbox[2] / factor)
    start = right + 10
    if image.width - start < image.width * 0.35:
        return []
    split = int(start + (image.width - start) * 0.47)
    return [image.crop((start, 0, split, image.height)),
            image.crop((split, 0, image.width, image.height))]


def read_clues(images: list[bytes], puzzle: Puzzle, layout_page: int | None,
               detection: GridDetection | None, document_text: str | None = None) -> tuple[dict, str] | None:
    """Return complete, numbered clues or defer to the vision reader."""
    if document_text:
        clues = _align(_parse(document_text), puzzle)
        if clues is not None:
            return clues, "PDF text"
    if not available():
        return None
    try:
        pages = [Image.open(io.BytesIO(data)).convert("RGB") for data in images]
        if detection is not None and layout_page is not None:
            crops = _side_by_side(pages[layout_page - 1], detection)
            if crops:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    texts = list(pool.map(lambda crop: _recognize(crop, 6, upscale=True), crops))
                combined = "\n\n".join(texts)
                clues = _align(_parse(combined), puzzle)
                if clues is not None:
                    return clues, "Tesseract OCR"
                # Small print sometimes survives at its original pixel size.
                with ThreadPoolExecutor(max_workers=2) as pool:
                    texts = list(pool.map(lambda crop: _recognize(crop, 6), crops))
                clues = _align(_parse("\n\n".join(texts)), puzzle)
                if clues is not None:
                    return clues, "Tesseract OCR"
        # Clues may be below the grid or on separate selected PDF pages.
        with ThreadPoolExecutor(max_workers=min(4, len(pages))) as pool:
            texts = list(pool.map(lambda page: _recognize(page, 3), pages))
        clues = _align(_parse("\n\n".join(texts)), puzzle)
        if clues is not None:
            return clues, "Tesseract OCR"
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError):
        return None
    return None
