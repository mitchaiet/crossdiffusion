"""Parse .xd crossword files (https://github.com/century-arcade/xd) into Puzzle objects."""

from __future__ import annotations

import re
from pathlib import Path

from crossword_agent.grid import Puzzle

_CLUE = re.compile(r"^([AD])(\d+)\.\s*(.*?)\s*~\s*(.*)$")
_MARKUP = re.compile(r"\{[/*_-](.*?)[/*_-]\}")


def parse_xd(text: str, source: str = "") -> Puzzle | None:
    """Return a Puzzle, or None if the file uses features we do not model (rebus, odd cells)."""
    text = text.replace("\r\n", "\n")
    # explicit section headers or blank-line separated blocks
    if "## " in text:
        sections: dict[str, str] = {}
        for block in re.split(r"^## ", text, flags=re.M):
            if not block.strip():
                continue
            name, _, body = block.partition("\n")
            sections[name.strip().lower()] = body
        meta_txt = sections.get("metadata", "")
        grid_txt = sections.get("grid", "")
        clues_txt = sections.get("clues", "")
        notes_txt = sections.get("notes", "")
    else:
        blocks = [b for b in re.split(r"\n{3,}", text.strip()) if b.strip()]
        if len(blocks) < 3:
            return None
        meta_txt, grid_txt, clues_txt = blocks[0], blocks[1], blocks[2]
        notes_txt = blocks[3] if len(blocks) > 3 else ""

    meta: dict[str, str] = {}
    for line in meta_txt.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()
    if "rebus" in meta or "special" in meta:
        return None  # rebus / shaded cells: skip for the letter-canvas model

    rows = [r.strip() for r in grid_txt.strip().splitlines() if r.strip()]
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        return None
    norm = []
    for r in rows:
        out = ""
        for ch in r:
            if ch == "#":
                out += "#"
            elif ch.isalpha():
                out += ch.upper()
            else:
                return None  # digits (rebus), underscores (missing cells), etc.
        norm.append(out)

    clues: dict[str, dict[str, str]] = {"across": {}, "down": {}}
    for line in clues_txt.splitlines():
        m = _CLUE.match(line.strip())
        if not m:
            continue
        d, num, clue, _answer = m.groups()
        clue = _MARKUP.sub(r"\1", clue).strip()
        clues["across" if d == "A" else "down"][num] = clue

    try:
        puzzle = Puzzle.from_rows(
            norm, clues, title=meta.get("title", ""), author=meta.get("author", ""),
            notes=notes_txt.strip()[:500], source=source,
        )
    except ValueError:
        return None
    if puzzle.solution is None or not puzzle.slots:
        return None
    if any(not s.clue for s in puzzle.slots):
        return None  # numbering mismatch or missing clues
    if any(s.length < 3 for s in puzzle.slots):
        return None
    return puzzle


def load_xd_dir(root: Path) -> list[Puzzle]:
    out = []
    for path in sorted(root.rglob("*.xd")):
        p = parse_xd(path.read_text(encoding="utf-8", errors="replace"), source=str(path))
        if p is not None:
            out.append(p)
    return out
