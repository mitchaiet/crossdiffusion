"""Client side of the CrossDiffusion filler, and its integration with the agent.

Three ways to use the learned model:
  * `DiffusionSolver`   pure sampler: the whole grid in one iterative denoising run
  * `DiffusionOracle`   the filler as an *oracle* inside the agent loop: every round it
                        refills the grid conditioned on the letters the agent has committed,
                        and its words enter the candidate table next to the LLM's
  * `CompositeOracle`   LLM oracle + diffusion oracle together (the hybrid)
"""

from __future__ import annotations

import json
import time
import urllib.request

from .agent import SolveResult
from .grid import Grid, Puzzle
from .oracle import Candidate, ClueQuery, Judgement, JudgeQuery, Oracle, OracleStats


class DiffusionClient:
    def __init__(self, url: str = "http://100.106.68.91:8009", steps: int = 12, timeout: float = 600.0,
                 remask_frac: float = 0.0, samples: int = 1, temperature: float = 0.0) -> None:
        self.url = url.rstrip("/")
        self.steps = steps
        self.timeout = timeout
        self.remask_frac = remask_frac
        self.samples = samples
        self.temperature = temperature
        self.calls = 0
        self.elapsed_s = 0.0

    def fill(self, puzzle: Puzzle, known: dict, snapshots: bool = False) -> dict:
        rows = []
        for r in range(puzzle.height):
            rows.append("".join(
                "#" if (r, c) in puzzle.blocks else (known.get((r, c)) or ".") for c in range(puzzle.width)
            ))
        clues = {
            "across": {str(s.number): s.clue for s in puzzle.slots if s.direction == "A"},
            "down": {str(s.number): s.clue for s in puzzle.slots if s.direction == "D"},
        }
        body = json.dumps({"grid": rows, "clues": clues, "title": puzzle.title, "steps": self.steps,
                           "snapshots": snapshots, "remask_frac": self.remask_frac, "samples": self.samples,
                           "temperature": self.temperature}).encode()
        req = urllib.request.Request(self.url + "/fill", data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        self.calls += 1
        self.elapsed_s += time.time() - t0
        if "error" in data:
            raise RuntimeError(data["error"])
        return data


def _parse_fill(puzzle: Puzzle, data: dict) -> tuple[dict, dict]:
    letters, conf = {}, {}
    for r, row in enumerate(data["grid"]):
        for c, ch in enumerate(row):
            if ch.isalpha():
                letters[(r, c)] = ch
    for key, p in data.get("confidence", {}).items():
        r, c = key.split(",")
        conf[(int(r), int(c))] = float(p)
    return letters, conf


def _word_confidence(puzzle: Puzzle, slot, conf: dict) -> float:
    ps = [conf.get(c, 0.0) for c in slot.cells]
    if not ps or min(ps) == 0.0:
        return 0.0
    prod = 1.0
    for p in ps:
        prod *= p
    return prod ** (1.0 / len(ps))  # geometric mean of cell confidences


class DiffusionSolver:
    """Pure learned solver: one denoising run over the whole canvas, optionally snapped onto the
    crossword vocabulary with the belief-propagation solver (`snap=True`, no LLM involved)."""

    def __init__(self, client: DiffusionClient, snap: bool = False, vocab=None, snap_config=None) -> None:
        self.client = client
        self.snap = snap
        self.vocab = vocab
        self.snap_config = snap_config

    def solve(self, puzzle: Puzzle, snapshots: bool = False) -> SolveResult:
        t0 = time.time()
        data = self.client.fill(puzzle, {}, snapshots=snapshots)
        self.last_snapshots = data.get("snapshots")
        self.last_cell_confidence = data.get("confidence", {})
        letters, conf = _parse_fill(puzzle, data)
        grid = Grid(puzzle)
        for cell, ch in letters.items():
            if cell in grid.letters:
                grid.letters[cell] = ch
        snap_info = None
        if self.snap:
            from .snap import Vocabulary, snap_grid

            if self.vocab is None:
                self.vocab = Vocabulary.load()
            grid, snap_info = snap_grid(puzzle, letters, conf, self.vocab, self.snap_config)
            conf = {c: (conf.get(c, 0.5) if grid.letters[c] == letters.get(c) else 0.6) for c in puzzle.white_cells}
        confidences = {s.id: _word_confidence(puzzle, s, conf) for s in puzzle.slots}
        return SolveResult(
            puzzle=puzzle, grid=grid, committed=grid, confidences=confidences, posteriors={},
            candidates={s.id: {grid.word(s): confidences[s.id]} for s in puzzle.slots if grid.word(s)},
            trace=[{"type": "diffusion", **h} for h in data.get("history", [])] + ([{"type": "snap", **snap_info}] if snap_info else []),
            rounds=len(data.get("history", [])), elapsed_s=time.time() - t0,
            oracle_stats={"server_elapsed_s": data.get("elapsed_s")}, usage=None,
            model="crossdiffusion" + ("+snap" if self.snap else ""),
        )


class DiffusionOracle:
    """The filler as an oracle: candidates come from refilling the current grid."""

    def __init__(self, client: DiffusionClient, min_conf: float = 0.7, trust: float = 0.5) -> None:
        """min_conf: filler words below this (geometric-mean cell confidence) are not offered at all.
        trust: multiplier applied to the filler's confidence before it competes with LLM candidates.
        A weak filler otherwise floods the candidate table with plausible-looking junk that belief
        propagation then reinforces (measured: the 1B warm-up filler dragged the agent from 92% to
        ~35% letters on mixed 15x15s when offered ungated)."""
        self.client = client
        self.min_conf = min_conf
        self.trust = trust
        self.stats = OracleStats()
        self._puzzle: Puzzle | None = None

    def bind(self, puzzle: Puzzle) -> DiffusionOracle:
        self._puzzle = puzzle
        return self

    def propose(self, queries: list[ClueQuery], *, k: int, context: str, source: str = "diffusion") -> dict[str, list[Candidate]]:
        puzzle = self._puzzle
        if puzzle is None:
            raise RuntimeError("DiffusionOracle.bind(puzzle) first")
        self.stats.queries += len(queries)
        known = {}
        for q in queries:  # patterns carry every committed letter of these slots
            slot = puzzle.slot_by_id[q.slot_id]
            for cell, ch in zip(slot.cells, q.pattern, strict=True):
                if ch != "_":
                    known[cell] = ch
        data = self.client.fill(puzzle, known)
        letters, conf = _parse_fill(puzzle, data)
        out: dict[str, list[Candidate]] = {}
        for q in queries:
            slot = puzzle.slot_by_id[q.slot_id]
            word = "".join(letters.get(c, "") for c in slot.cells)
            if len(word) != slot.length:
                out[q.slot_id] = []
                continue
            p = _word_confidence(puzzle, slot, {**conf, **{c: 1.0 for c in known}})
            self.stats.raw_candidates += 1
            out[q.slot_id] = [Candidate(word, max(0.01, p * self.trust), source)] if p >= self.min_conf else []
        return out

    def judge(self, queries: list[JudgeQuery], *, context: str) -> dict[str, Judgement]:
        return {}  # the filler has no notion of "is this a fair answer"; the LLM judges


class CompositeOracle:
    """Combine a primary (LLM) oracle with secondary ones.

    mode="inject":   every oracle's candidates enter the table (the filler competes with the LLM)
    mode="fallback": secondary candidates are used only for slots where the primary returned nothing,
                     so the filler can never overrule an LLM candidate, only fill a blank
    Judging goes to the first oracle that can judge."""

    def __init__(self, oracles: list[Oracle], mode: str = "inject") -> None:
        self.oracles = oracles
        self.mode = mode
        self.stats = oracles[0].stats
        self.client = getattr(oracles[0], "client", None)  # usage accounting (LLM first)

    def propose(self, queries, *, k: int, context: str, source: str = "llm") -> dict[str, list[Candidate]]:
        merged: dict[str, list[Candidate]] = {q.slot_id: [] for q in queries}
        primary = self.oracles[0].propose(queries, k=k, context=context, source=source)
        for sid, cands in primary.items():
            merged.setdefault(sid, []).extend(cands)
        for o in self.oracles[1:]:
            part = o.propose(queries, k=k, context=context, source=f"{source}-{o.__class__.__name__}")
            for sid, cands in part.items():
                if self.mode == "fallback" and merged.get(sid):
                    continue
                merged.setdefault(sid, []).extend(cands)
        return merged

    def judge(self, queries, *, context: str) -> dict[str, Judgement]:
        for o in self.oracles:
            verdicts = o.judge(queries, context=context)
            if verdicts:
                return verdicts
        return {}
