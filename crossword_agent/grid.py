"""Puzzle model: grid geometry, slot numbering, loading from JSON and .puz files.

A `Puzzle` is immutable puzzle *structure* (layout, clues, optional solution).
A `Grid` is the mutable *working state* the agent fills in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Direction = Literal["A", "D"]
Cell = tuple[int, int]

BLOCK = "#"
EMPTY = "."


@dataclass(frozen=True)
class Slot:
    """One numbered entry (an across or down answer)."""

    number: int
    direction: Direction
    cells: tuple[Cell, ...]
    clue: str

    @property
    def id(self) -> str:
        return f"{self.number}{self.direction}"

    @property
    def length(self) -> int:
        return len(self.cells)

    @property
    def start(self) -> Cell:
        return self.cells[0]

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.id} ({self.length}): {self.clue}"


@dataclass
class Puzzle:
    """A crossword: layout, clues, and (if known) the solution."""

    width: int
    height: int
    blocks: frozenset[Cell]
    slots: list[Slot]
    solution: dict[Cell, str] | None = None
    title: str = ""
    author: str = ""
    notes: str = ""
    difficulty: str = ""
    source: str = ""

    # derived indexes
    slot_by_id: dict[str, Slot] = field(init=False, repr=False)
    cell_slots: dict[Cell, list[tuple[str, int]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.slot_by_id = {s.id: s for s in self.slots}
        self.cell_slots = {}
        for s in self.slots:
            for i, c in enumerate(s.cells):
                self.cell_slots.setdefault(c, []).append((s.id, i))

    # ---- geometry -------------------------------------------------------
    @property
    def white_cells(self) -> list[Cell]:
        return [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.blocks
        ]

    def is_block(self, cell: Cell) -> bool:
        return cell in self.blocks

    def crossings(self, slot: Slot) -> list[tuple[int, Slot, int]]:
        """For each cell index in `slot`, the crossing slot and its index there."""
        out = []
        for i, cell in enumerate(slot.cells):
            for other_id, j in self.cell_slots[cell]:
                if other_id != slot.id:
                    out.append((i, self.slot_by_id[other_id], j))
        return out

    def solution_word(self, slot: Slot) -> str | None:
        if self.solution is None:
            return None
        return "".join(self.solution[c] for c in slot.cells)

    # ---- constructors ---------------------------------------------------
    @staticmethod
    def number_slots(
        width: int, height: int, blocks: set[Cell], clues: dict[str, dict[str, str]]
    ) -> list[Slot]:
        """Standard American-style numbering: a cell gets a number if it starts an
        across or down run of length >= 2."""
        slots: list[Slot] = []
        n = 0
        for r in range(height):
            for c in range(width):
                if (r, c) in blocks:
                    continue
                starts_across = (c == 0 or (r, c - 1) in blocks) and (
                    c + 1 < width and (r, c + 1) not in blocks
                )
                starts_down = (r == 0 or (r - 1, c) in blocks) and (
                    r + 1 < height and (r + 1, c) not in blocks
                )
                if not (starts_across or starts_down):
                    continue
                n += 1
                if starts_across:
                    cells = []
                    cc = c
                    while cc < width and (r, cc) not in blocks:
                        cells.append((r, cc))
                        cc += 1
                    slots.append(
                        Slot(n, "A", tuple(cells), clues.get("across", {}).get(str(n), ""))
                    )
                if starts_down:
                    cells = []
                    rr = r
                    while rr < height and (rr, c) not in blocks:
                        cells.append((rr, c))
                        rr += 1
                    slots.append(Slot(n, "D", tuple(cells), clues.get("down", {}).get(str(n), "")))
        return slots

    @classmethod
    def from_rows(
        cls,
        rows: list[str],
        clues: dict[str, dict[str, str]],
        *,
        title: str = "",
        author: str = "",
        notes: str = "",
        difficulty: str = "",
        source: str = "",
    ) -> Puzzle:
        """Build from row strings: '#' = block, letter = solution, '.' = unknown."""
        height = len(rows)
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("all rows must have the same width")
        blocks = {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == BLOCK}
        solution: dict[Cell, str] | None = {}
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == BLOCK:
                    continue
                if ch == EMPTY or not ch.isalpha():
                    solution = None
                    break
                solution[(r, c)] = ch.upper()
            if solution is None:
                break
        slots = cls.number_slots(width, height, blocks, clues)
        return cls(
            width=width,
            height=height,
            blocks=frozenset(blocks),
            slots=slots,
            solution=solution,
            title=title,
            author=author,
            notes=notes,
            difficulty=difficulty,
            source=source,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Puzzle:
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_rows(
            data["grid"],
            data.get("clues", {}),
            title=data.get("title", path.stem),
            author=data.get("author", ""),
            notes=data.get("notes", ""),
            difficulty=data.get("difficulty", ""),
            source=str(path),
        )

    @classmethod
    def from_puz(cls, path: str | Path) -> Puzzle:
        """Load an AcrossLite .puz file (via the `puzpy` package)."""
        import puz  # lazy import; optional dependency at runtime

        p = puz.read(str(path))
        numbering = p.clue_numbering()
        rows: list[str] = []
        locked = p.is_solution_locked()
        for r in range(p.height):
            row = ""
            for c in range(p.width):
                ch = p.solution[r * p.width + c]
                if ch == ".":
                    row += BLOCK
                elif locked or not ch.isalpha():
                    row += EMPTY
                else:
                    row += ch.upper()
            rows.append(row)
        clues = {
            "across": {str(e["num"]): e["clue"] for e in numbering.across},
            "down": {str(e["num"]): e["clue"] for e in numbering.down},
        }
        return cls.from_rows(
            rows,
            clues,
            title=(p.title or Path(path).stem).strip(),
            author=(p.author or "").strip(),
            notes=(p.notes or "").strip(),
            source=str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> Puzzle:
        path = Path(path)
        if path.suffix.lower() == ".puz":
            return cls.from_puz(path)
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        raise ValueError(f"unsupported puzzle format: {path.suffix}")

    # ---- export ---------------------------------------------------------
    def to_dict(self, include_solution: bool = True) -> dict:
        rows = []
        for r in range(self.height):
            row = ""
            for c in range(self.width):
                if (r, c) in self.blocks:
                    row += BLOCK
                elif include_solution and self.solution:
                    row += self.solution[(r, c)]
                else:
                    row += EMPTY
            rows.append(row)
        return {
            "title": self.title,
            "author": self.author,
            "difficulty": self.difficulty,
            "notes": self.notes,
            "grid": rows,
            "clues": {
                "across": {str(s.number): s.clue for s in self.slots if s.direction == "A"},
                "down": {str(s.number): s.clue for s in self.slots if s.direction == "D"},
            },
        }


class Grid:
    """Mutable working fill for a puzzle."""

    def __init__(self, puzzle: Puzzle) -> None:
        self.puzzle = puzzle
        self.letters: dict[Cell, str | None] = {c: None for c in puzzle.white_cells}

    def copy(self) -> Grid:
        g = Grid(self.puzzle)
        g.letters = dict(self.letters)
        return g

    def pattern(self, slot: Slot, unknown: str = "_") -> str:
        return "".join(self.letters[c] or unknown for c in slot.cells)

    def word(self, slot: Slot) -> str | None:
        """The full word in the slot, or None if any cell is empty."""
        letters = [self.letters[c] for c in slot.cells]
        if any(ch is None for ch in letters):
            return None
        return "".join(letters)  # type: ignore[arg-type]

    def fits(self, slot: Slot, word: str) -> bool:
        if len(word) != slot.length:
            return False
        return all(self.letters[c] in (None, ch) for c, ch in zip(slot.cells, word, strict=True))

    def set_word(self, slot: Slot, word: str) -> None:
        if len(word) != slot.length:
            raise ValueError(f"{word!r} does not fit {slot.id} (len {slot.length})")
        for c, ch in zip(slot.cells, word, strict=True):
            self.letters[c] = ch

    def clear_slot(self, slot: Slot, keep: set[Cell] | None = None) -> None:
        for c in slot.cells:
            if keep and c in keep:
                continue
            self.letters[c] = None

    def filled_fraction(self) -> float:
        total = len(self.letters)
        return sum(1 for v in self.letters.values() if v) / total if total else 0.0

    def is_complete(self) -> bool:
        return all(v is not None for v in self.letters.values())

    def rows(self, empty: str = EMPTY) -> list[str]:
        out = []
        for r in range(self.puzzle.height):
            row = ""
            for c in range(self.puzzle.width):
                if (r, c) in self.puzzle.blocks:
                    row += BLOCK
                else:
                    row += self.letters[(r, c)] or empty
            out.append(row)
        return out

    def __str__(self) -> str:
        return "\n".join(" ".join(row) for row in self.rows())
