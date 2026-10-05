"""Render a puzzle as a newspaper-style PNG (grid with numbers + clue columns), for testing the
screenshot reader and for demos.

    .venv/bin/python scripts/render_puzzle_png.py puzzles/medium/medium_03.json out.png [--solution]
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from crossword_agent.evaluate import load_puzzle_set
from crossword_agent.grid import Puzzle


def _font(size: int):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(puzzle: Puzzle, out: Path, cell: int = 44, solution: bool = False, col_width: int = 340) -> None:
    gw, gh = puzzle.width * cell, puzzle.height * cell
    across = [s for s in puzzle.slots if s.direction == "A"]
    down = [s for s in puzzle.slots if s.direction == "D"]
    small, body, head = _font(11), _font(15), _font(20)
    wrap = 38
    lines_a = [(f"{s.number}. {s.clue}") for s in across]
    lines_d = [(f"{s.number}. {s.clue}") for s in down]
    def wrapped(lines):
        out_lines = []
        for ln in lines:
            out_lines += textwrap.wrap(ln, wrap) or [ln]
        return out_lines
    la, ld = wrapped(lines_a), wrapped(lines_d)
    text_h = max(len(la), len(ld)) * 19 + 60
    W = 40 + gw + 40 + col_width * 2 + 40
    H = max(gh + 80, text_h + 40)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 20), puzzle.title or "Crossword", fill="black", font=head)
    ox, oy = 40, 60
    for r in range(puzzle.height):
        for c in range(puzzle.width):
            x0, y0 = ox + c * cell, oy + r * cell
            if (r, c) in puzzle.blocks:
                d.rectangle([x0, y0, x0 + cell, y0 + cell], fill="black")
            else:
                d.rectangle([x0, y0, x0 + cell, y0 + cell], outline="black", fill="white")
                if solution and puzzle.solution:
                    d.text((x0 + cell / 2 - 6, y0 + 12), puzzle.solution[(r, c)], fill="black", font=head)
    for s in puzzle.slots:
        r, c = s.start
        d.text((ox + c * cell + 3, oy + r * cell + 2), str(s.number), fill="black", font=small)
    tx = ox + gw + 40
    for col, (title, lines) in enumerate((("ACROSS", la), ("DOWN", ld))):
        x = tx + col * col_width
        d.text((x, 60), title, fill="black", font=head)
        for i, ln in enumerate(lines):
            d.text((x, 95 + i * 19), ln, fill="black", font=body)
    img.save(out)
    print(f"wrote {out} ({W}x{H})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()
    puzzle = load_puzzle_set([args.puzzle], limit=args.index + 1)[args.index][1]
    render(puzzle, args.out, solution=args.solution)


if __name__ == "__main__":
    main()
