"""Constraint solver: turns per-clue candidate lists into a consistent grid.

Three pieces, used by the agent in a loop:

1. `propagate`   loopy belief propagation over the grid graph. Each slot has a
                  distribution over its candidate words (plus an explicit "none of
                  these" mass). Letter marginals flow through crossing cells, so a
                  candidate whose letters agree with confident crossings is boosted
                  and one that disagrees is suppressed.
2. `choose_commits` picks the entries the solver is sure enough about to write down.
3. `backtrack_fill` finishes the remaining slots with weighted DFS (max total
                  log-probability, node-limited, anytime), allowing a slot to stay
                  open when nothing plausible fits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .grid import Grid, Puzzle, Slot
from .oracle import Candidate

# English letter frequencies (approximate), used as the background distribution
# for the "answer is not among the candidates" hypothesis.
_FREQ = {
    "E": 12.7, "T": 9.1, "A": 8.2, "O": 7.5, "I": 7.0, "N": 6.7, "S": 6.3, "H": 6.1,
    "R": 6.0, "D": 4.3, "L": 4.0, "C": 2.8, "U": 2.8, "M": 2.4, "W": 2.4, "F": 2.2,
    "G": 2.0, "Y": 2.0, "P": 1.9, "B": 1.5, "V": 1.0, "K": 0.8, "J": 0.15, "X": 0.15,
    "Q": 0.10, "Z": 0.07,
}
_TOTAL = sum(_FREQ.values())
LETTER_FREQ = {k: v / _TOTAL for k, v in _FREQ.items()}

OTHER = "\x00"  # pseudo-candidate key: "the answer is none of the listed candidates"

CandidateTable = dict[str, dict[str, float]]  # slot_id -> word -> prior confidence
Posteriors = dict[str, dict[str, float]]  # slot_id -> word (or OTHER) -> probability


@dataclass
class SolverConfig:
    bp_iterations: int = 3
    damping: float = 0.3  # exponent on crossing evidence; simulated sweep on 40 real grids peaks at 0.2-0.3 (0.0 = no crossings, much worse)
    other_floor: float = 0.05  # minimum mass for "none of the candidates"
    other_ceiling: float = 0.97
    commit_threshold: float = 0.8  # posterior needed to write an entry into the grid
    commit_margin: float = 0.3  # gap to the runner-up (or to OTHER) needed to commit
    backtrack_nodes: int = 100_000


# --------------------------------------------------------------------------------------
# candidate bookkeeping
# --------------------------------------------------------------------------------------
def merge_candidates(table: CandidateTable, new: dict[str, list[Candidate]]) -> int:
    """Union new candidates into the table (keeping the max confidence). Returns #new words."""
    added = 0
    for slot_id, cands in new.items():
        bucket = table.setdefault(slot_id, {})
        for c in cands:
            if c.answer not in bucket:
                added += 1
                bucket[c.answer] = c.confidence
            else:
                bucket[c.answer] = max(bucket[c.answer], c.confidence)
    return added


def _prior(cands: dict[str, float], cfg: SolverConfig) -> dict[str, float]:
    """Turn raw confidences into a distribution that includes OTHER.

    OTHER takes 1 - (top confidence), clamped to [floor, ceiling]; the rest is shared among the
    candidates in proportion to their confidence. The ceiling must be close to 1: with a low
    ceiling a lone junk guess offered at 0.02 inherited ~40% of the mass and, through crossings,
    propped up other junk instead of leaving the slot open for a better-informed re-ask."""
    if not cands:
        return {OTHER: 1.0}
    total = sum(cands.values())
    p_other = min(cfg.other_ceiling, max(cfg.other_floor, 1.0 - max(cands.values())))
    dist = {w: (1.0 - p_other) * c / total for w, c in cands.items()}
    dist[OTHER] = p_other
    return dist


# --------------------------------------------------------------------------------------
# belief propagation
# --------------------------------------------------------------------------------------
def _letter_messages(
    puzzle: Puzzle, grid: Grid, dists: Posteriors
) -> dict[tuple[str, int], dict[str, float]]:
    """For every (slot, index): a distribution over letters at that cell implied by the slot."""
    msgs: dict[tuple[str, int], dict[str, float]] = {}
    for slot in puzzle.slots:
        dist = dists.get(slot.id, {OTHER: 1.0})
        p_other = dist.get(OTHER, 0.0)
        for i, cell in enumerate(slot.cells):
            fixed = grid.letters[cell]
            if fixed:
                msgs[(slot.id, i)] = {fixed: 1.0}
                continue
            m = {L: p_other * f for L, f in LETTER_FREQ.items()}
            for w, p in dist.items():
                if w == OTHER:
                    continue
                m[w[i]] = m.get(w[i], 0.0) + p
            msgs[(slot.id, i)] = m
    return msgs


def propagate(puzzle: Puzzle, grid: Grid, table: CandidateTable, cfg: SolverConfig) -> Posteriors:
    """Run damped loopy BP; return per-slot posteriors over candidates (+ OTHER)."""
    priors: Posteriors = {}
    for slot in puzzle.slots:
        fitting = {w: c for w, c in table.get(slot.id, {}).items() if grid.fits(slot, w)}
        priors[slot.id] = _prior(fitting, cfg)

    dists = dict(priors)
    crossings = {s.id: puzzle.crossings(s) for s in puzzle.slots}
    for _ in range(cfg.bp_iterations):
        msgs = _letter_messages(puzzle, grid, dists)
        new: Posteriors = {}
        for slot in puzzle.slots:
            prior = priors[slot.id]
            logs: dict[str, float] = {}
            for w, p in prior.items():
                score = math.log(max(p, 1e-12))
                if w != OTHER:
                    for i, other, j in crossings[slot.id]:
                        m = msgs[(other.id, j)]
                        letter = w[i]
                        evidence = m.get(letter, 0.0) / LETTER_FREQ[letter]
                        score += cfg.damping * math.log(max(evidence, 1e-9))
                logs[w] = score
            mx = max(logs.values())
            z = sum(math.exp(v - mx) for v in logs.values())
            new[slot.id] = {w: math.exp(v - mx) / z for w, v in logs.items()}
        dists = new
    return dists


# --------------------------------------------------------------------------------------
# committing
# --------------------------------------------------------------------------------------
def choose_commits(
    puzzle: Puzzle, grid: Grid, posteriors: Posteriors, cfg: SolverConfig
) -> list[tuple[Slot, str, float]]:
    """Entries confident enough to write into the grid, mutually consistent, best first."""
    proposals: list[tuple[float, Slot, str]] = []
    for slot in puzzle.slots:
        if grid.word(slot) is not None:
            continue
        dist = posteriors.get(slot.id, {})
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][0] == OTHER:
            continue
        best_word, p1 = ranked[0]
        p2 = ranked[1][1] if len(ranked) > 1 else 0.0
        if p1 >= cfg.commit_threshold and (p1 - p2) >= cfg.commit_margin:
            proposals.append((p1, slot, best_word))
    proposals.sort(key=lambda t: -t[0])
    scratch = grid.copy()
    commits: list[tuple[Slot, str, float]] = []
    for p, slot, word in proposals:
        if scratch.fits(slot, word):
            scratch.set_word(slot, word)
            commits.append((slot, word, p))
    return commits


# --------------------------------------------------------------------------------------
# backtracking completion
# --------------------------------------------------------------------------------------
class _Budget(Exception):
    pass


def backtrack_fill(
    puzzle: Puzzle, grid: Grid, posteriors: Posteriors, cfg: SolverConfig
) -> tuple[Grid, float, int]:
    """Fill open slots maximising total log posterior. Returns (grid, score, nodes)."""
    ranked: dict[str, list[tuple[str, float]]] = {}
    other_lp: dict[str, float] = {}
    for slot in puzzle.slots:
        dist = posteriors.get(slot.id, {OTHER: 1.0})
        words = sorted(
            ((w, math.log(max(p, 1e-12))) for w, p in dist.items() if w != OTHER),
            key=lambda t: -t[1],
        )
        ranked[slot.id] = words
        other_lp[slot.id] = math.log(max(dist.get(OTHER, cfg.other_floor), 1e-12))

    def slot_value(g: Grid, slot: Slot) -> float:
        """Score contribution of a slot that is now full in g."""
        w = g.word(slot)
        for cand, lp in ranked[slot.id]:
            if cand == w:
                return lp
        return other_lp[slot.id]  # formed by crossings only: treat as "not a listed candidate"

    best: dict = {"score": -math.inf, "grid": grid.copy()}
    state = {"nodes": 0}
    limit = cfg.backtrack_nodes

    def rec(g: Grid, remaining: set[str], score: float) -> None:
        state["nodes"] += 1
        if state["nodes"] > limit:
            raise _Budget
        # absorb slots completed by crossings
        done = [sid for sid in remaining if g.word(puzzle.slot_by_id[sid]) is not None]
        for sid in done:
            score += slot_value(g, puzzle.slot_by_id[sid])
            remaining = remaining - {sid}
        # optimistic bound: every remaining slot could at best add its top log-prob (<= 0)
        bound = score + sum(
            max([lp for _, lp in ranked[sid][:1]] + [other_lp[sid]]) for sid in remaining
        )
        if bound <= best["score"]:
            return
        if not remaining:
            if score > best["score"]:
                best["score"], best["grid"] = score, g.copy()
            return
        # most constrained slot first
        def n_fits(sid: str) -> int:
            s = puzzle.slot_by_id[sid]
            return sum(1 for w, _ in ranked[sid] if g.fits(s, w))

        sid = min(remaining, key=lambda s: (n_fits(s), -len(puzzle.slot_by_id[s].cells)))
        slot = puzzle.slot_by_id[sid]
        rest = remaining - {sid}
        for w, lp in ranked[sid]:
            if g.fits(slot, w):
                g2 = g.copy()
                g2.set_word(slot, w)
                rec(g2, rest, score + lp)
        # leave it open (letters may still arrive from crossings)
        rec(g, rest, score + other_lp[sid])

    open_slots = {s.id for s in puzzle.slots if grid.word(s) is None}
    base = sum(slot_value(grid, s) for s in puzzle.slots if grid.word(s) is not None)
    try:
        rec(grid.copy(), open_slots, base)
    except _Budget:
        pass
    if best["score"] == -math.inf:
        return grid.copy(), base, state["nodes"]
    return best["grid"], best["score"], state["nodes"]


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def slot_confidences(puzzle: Puzzle, grid: Grid, posteriors: Posteriors) -> dict[str, float]:
    """Posterior probability of the word actually written in each slot (0 if unlisted/open)."""
    out: dict[str, float] = {}
    for slot in puzzle.slots:
        w = grid.word(slot)
        out[slot.id] = posteriors.get(slot.id, {}).get(w, 0.0) if w else 0.0
    return out
