"""Baselines the agent is compared against.

oneshot   one prompt: the whole puzzle in, a full answer list out (no constraint reasoning)
top1      one candidate pass with empty patterns, write the top guess per clue, no propagation
"""

from __future__ import annotations

import json
import time

from .agent import SolveResult
from .grid import Grid, Puzzle
from .llm import LLMError
from .oracle import ClueQuery, LLMOracle, Oracle, normalize_answer

ONESHOT_SYSTEM = """You are an expert solver of American-style crossword puzzles. You get the grid
layout ('#' = black square, '.' = empty) and all clues with answer lengths. Solve the whole puzzle.
Answers are uppercase A-Z only (no spaces or punctuation) and must have the stated length. Across and
down answers must agree on shared squares. You MUST return an answer for every clue id: when unsure,
give your best guess of the right length rather than omitting it. Never return an empty object.
Respond with JSON only: {"answers": {"1A": "WORD", "1D": "WORD", ...}}"""


def _fill_ordered(puzzle: Puzzle, words: dict[str, str]) -> tuple[Grid, int]:
    """Write across answers then down answers, never overwriting; count conflicts."""
    grid = Grid(puzzle)
    conflicts = 0
    for direction in ("A", "D"):
        for s in puzzle.slots:
            if s.direction != direction:
                continue
            w = words.get(s.id)
            if not w or len(w) != s.length:
                continue
            for cell, ch in zip(s.cells, w, strict=True):
                if grid.letters[cell] is None:
                    grid.letters[cell] = ch
                elif grid.letters[cell] != ch:
                    conflicts += 1
    return grid, conflicts


def solve_oneshot(client, puzzle: Puzzle) -> SolveResult:
    t0 = time.time()
    layout = [
        "".join("#" if (r, c) in puzzle.blocks else "." for c in range(puzzle.width))
        for r in range(puzzle.height)
    ]
    clues = [{"id": s.id, "clue": s.clue, "length": s.length} for s in puzzle.slots]
    user = (
        f"Puzzle: {puzzle.title} ({puzzle.width}x{puzzle.height})\nLayout:\n" + "\n".join(layout)
        + "\n\nClues:\n" + json.dumps(clues, ensure_ascii=False)
    )
    error = None
    try:
        data = client.complete_json(ONESHOT_SYSTEM, user)
    except LLMError as e:  # a single-call baseline has no smaller unit to fall back to
        data, error = {}, repr(e)
    raw = data.get("answers", {}) if isinstance(data, dict) else {}
    retried = False
    if isinstance(raw, list):  # some models return [{"id":..,"answer":..}]
        raw = {str(e.get("id")): e.get("answer", "") for e in raw if isinstance(e, dict)}
    if len(raw) < len(puzzle.slots) // 2 and error is None:  # gave up or truncated: one more try, uncached
        retried = True
        try:
            data = client.complete_json(ONESHOT_SYSTEM, user, nocache=True)
        except LLMError as e:
            data, error = {}, repr(e)
        raw2 = data.get("answers", {}) if isinstance(data, dict) else {}
        if isinstance(raw2, list):
            raw2 = {str(e.get("id")): e.get("answer", "") for e in raw2 if isinstance(e, dict)}
        if len(raw2) > len(raw):
            raw = raw2
    words = {str(k).upper().replace(" ", ""): normalize_answer(v) for k, v in raw.items()}
    # models that give up return placeholders ("XXXXXX", "??????"); count them, do not write them
    placeholders = {k for k, w in words.items() if not w or len(set(w)) == 1 or "?" in str(raw.get(k, ""))}
    words = {k: w for k, w in words.items() if k not in placeholders}
    grid, conflicts = _fill_ordered(puzzle, words)
    conf = {s.id: (0.5 if grid.word(s) else 0.0) for s in puzzle.slots}
    return SolveResult(
        puzzle=puzzle, grid=grid, committed=grid, confidences=conf, posteriors={},
        candidates={sid: {w: 0.5} for sid, w in words.items() if w},
        trace=[{"type": "oneshot", "conflicts": conflicts, "answers": len(words),
                "placeholders": len(placeholders), "retried": retried, "error": error}],
        rounds=1, elapsed_s=time.time() - t0, oracle_stats={},
        usage=client.usage.to_dict(), model=client.config.model,
    )


def solve_top1(oracle: Oracle, puzzle: Puzzle, k: int = 5) -> SolveResult:
    t0 = time.time()
    queries = [ClueQuery(s.id, s.clue, s.length, "_" * s.length) for s in puzzle.slots]
    cands = oracle.propose(queries, k=k, context=f"Puzzle: {puzzle.title}")
    grid = Grid(puzzle)
    ranked = sorted(
        ((c[0].confidence, sid, c[0].answer) for sid, c in cands.items() if c), reverse=True
    )
    conf: dict[str, float] = {s.id: 0.0 for s in puzzle.slots}
    for p, sid, word in ranked:
        slot = puzzle.slot_by_id[sid]
        if grid.fits(slot, word):
            grid.set_word(slot, word)
            conf[sid] = p
    client = getattr(oracle, "client", None)
    return SolveResult(
        puzzle=puzzle, grid=grid, committed=grid, confidences=conf, posteriors={},
        candidates={sid: {c.answer: c.confidence for c in cs} for sid, cs in cands.items()},
        trace=[{"type": "top1", "written": sum(1 for v in conf.values() if v)}],
        rounds=1, elapsed_s=time.time() - t0, oracle_stats=oracle.stats.to_dict(),
        usage=client.usage.to_dict() if client else None,
        model=client.config.model if isinstance(oracle, LLMOracle) else None,
    )
