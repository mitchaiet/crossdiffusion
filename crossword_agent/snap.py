"""Vocabulary snapping: turn CrossDiffusion's letter grid into a grid of real crossword answers.

The filler's mistakes are mostly one or two letters inside an otherwise-right word. For every
slot we look up vocabulary words within a small Hamming distance of what the filler wrote, score
them by how confident the filler was in the letters they keep, how common the word is in
published crosswords, and then let the agent's belief-propagation solver choose a mutually
consistent set. No language model is involved.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .grid import Grid, Puzzle
from .solver import SolverConfig, backtrack_fill, choose_commits, propagate

DEFAULT_VOCAB = Path(__file__).resolve().parents[1] / "data" / "corpus" / "vocab.json"
DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "data" / "corpus" / "crossdiffusion_calibration.json"


def load_calibration(path: Path | str = DEFAULT_CALIBRATION) -> dict | None:
    """Confidence bin -> empirical accuracy, measured on held-out fills (see docs/DIFFUSION.md)."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@dataclass
class SnapConfig:
    max_hamming: int = 2  # +1 for words longer than 6 letters
    keep_original_prior: float = 0.05  # weight of the filler's own word when it is not in the vocabulary
    conf_floor: float = 0.2  # clamp (calibrated) cell confidences into [floor, ceil]
    conf_ceil: float = 0.9
    freq_weight: float = 0.12  # per log10(usage count)
    mismatch_spread: float = 3.0  # a differing letter costs (1-p)/spread; the vocabulary decides which letter
    top_k: int = 12
    commit_threshold: float = 0.5
    calibration: dict | None = None  # confidence bin (0-9) -> empirical accuracy


class Vocabulary:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.by_len: dict[int, list[str]] = defaultdict(list)
        self.index: dict[tuple[int, int, str], set[str]] = defaultdict(set)
        for w in counts:
            self.by_len[len(w)].append(w)
            for i, ch in enumerate(w):
                self.index[(len(w), i, ch)].add(w)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_VOCAB) -> Vocabulary:
        return cls(json.loads(Path(path).read_text()))

    def near(self, word: str, max_hamming: int) -> list[tuple[str, int]]:
        """Vocabulary words of the same length within `max_hamming` substitutions, with distance."""
        L = len(word)
        need = L - max_hamming
        tally: dict[str, int] = defaultdict(int)
        for i, ch in enumerate(word):
            for w in self.index.get((L, i, ch), ()):
                tally[w] += 1
        return [(w, L - m) for w, m in tally.items() if m >= need]


def snap_grid(puzzle: Puzzle, letters: dict, confidence: dict, vocab: Vocabulary,
              cfg: SnapConfig | None = None) -> tuple[Grid, dict]:
    cfg = cfg or SnapConfig()
    if cfg.calibration is None:
        cfg.calibration = load_calibration()
    table: dict[str, dict[str, float]] = {}
    for slot in puzzle.slots:
        w0 = "".join(letters.get(c) or "?" for c in slot.cells)
        if "?" in w0:
            continue
        raw = [confidence.get(c, 0.5) for c in slot.cells]
        if cfg.calibration:
            raw = [cfg.calibration.get(str(min(9, int(v * 10))), cfg.calibration.get(min(9, int(v * 10)), v)) for v in raw]
        p = [min(cfg.conf_ceil, max(cfg.conf_floor, v)) for v in raw]
        limit = cfg.max_hamming + (1 if slot.length > 6 else 0)
        scores: dict[str, float] = {}
        for w, _dist in vocab.near(w0, limit):
            s = 0.0
            for i, ch in enumerate(w):
                s += math.log(p[i]) if ch == w0[i] else math.log((1 - p[i]) / cfg.mismatch_spread)
            s += cfg.freq_weight * math.log10(vocab.counts[w] + 1)
            scores[w] = s
        if w0 not in scores:  # the filler's word is not a known answer: keep it as a weak option
            scores[w0] = sum(math.log(pi) for pi in p) + math.log(cfg.keep_original_prior)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[: cfg.top_k]
        mx = top[0][1]
        z = sum(math.exp(v - mx) for _, v in top)
        table[slot.id] = {w: 0.95 * math.exp(v - mx) / z for w, v in top}

    solver_cfg = SolverConfig(commit_threshold=cfg.commit_threshold, commit_margin=0.15)
    grid = Grid(puzzle)
    post = propagate(puzzle, grid, table, solver_cfg)
    committed = 0
    for slot, word, _p in choose_commits(puzzle, grid, post, solver_cfg):
        grid.set_word(slot, word)
        committed += 1
    post = propagate(puzzle, grid, table, solver_cfg)
    grid, _score, nodes = backtrack_fill(puzzle, grid, post, solver_cfg)
    restored = 0
    for c in puzzle.white_cells:  # anything the solver left open keeps the filler's letter
        if grid.letters[c] is None and letters.get(c):
            grid.letters[c] = letters[c]
            restored += 1
    changed = sum(1 for c in puzzle.white_cells if letters.get(c) and grid.letters[c] != letters[c])
    return grid, {"committed": committed, "search_nodes": nodes, "restored": restored, "changed_cells": changed}
