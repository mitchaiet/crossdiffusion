"""The crossword agent: an observe -> propose -> reason -> commit -> verify loop.

Round structure
    1. observe   read the grid; every open slot yields a (clue, length, pattern) query
    2. propose   the oracle (LLM) returns ranked candidates for the patterns not yet asked
    3. reason    belief propagation across crossings turns per-clue guesses into
                 grid-consistent posteriors
    4. commit    confident, mutually consistent entries are written into the grid,
                 which sharpens the patterns for the next round
After the rounds: weighted backtracking completes the grid, a judge pass asks the LLM
to check weak entries (and un-commits ones it rejects), and a last force-fill pass
takes best guesses for anything still open.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .grid import Grid, Puzzle, Slot
from .oracle import ClueQuery, JudgeQuery, Oracle
from .solver import (
    OTHER,
    CandidateTable,
    Posteriors,
    SolverConfig,
    backtrack_fill,
    choose_commits,
    merge_candidates,
    propagate,
    slot_confidences,
)

# ``stage`` events describe work about to begin, before a potentially slow model
# call or solver operation. They expose progress, never model reasoning. Other
# events describe completed work and may include newly committed grid letters.
EventCallback = Callable[[dict, Grid], None]


@dataclass
class AgentConfig:
    k: int = 5  # candidates requested per clue
    max_rounds: int = 4
    threshold_decay: float = 0.05  # commit threshold relaxes by this much each round
    min_threshold: float = 0.55
    repair_cycles: int = 1
    judge_below: float = 0.5  # entries with posterior below this get judged
    loosen_crossings_below: float = 1.01  # on a rejection, un-commit its crossings too (their words stay as candidates)
    force_fill: bool = True
    solver: SolverConfig = field(default_factory=SolverConfig)


@dataclass
class SolveResult:
    puzzle: Puzzle
    grid: Grid
    committed: Grid
    confidences: dict[str, float]
    posteriors: Posteriors
    candidates: CandidateTable
    trace: list[dict]
    rounds: int
    elapsed_s: float
    oracle_stats: dict
    usage: dict | None = None
    model: str | None = None

    def answers(self) -> dict[str, str | None]:
        return {s.id: self.grid.word(s) for s in self.puzzle.slots}

    def to_dict(self) -> dict:
        return {
            "title": self.puzzle.title,
            "source": self.puzzle.source,
            "model": self.model,
            "rounds": self.rounds,
            "elapsed_s": round(self.elapsed_s, 2),
            "grid": self.grid.rows(),
            "answers": self.answers(),
            "confidences": {k: round(v, 4) for k, v in self.confidences.items()},
            "oracle_stats": self.oracle_stats,
            "usage": self.usage,
            "trace": self.trace,
            "candidates": {sid: dict(sorted(c.items(), key=lambda kv: -kv[1])[:8]) for sid, c in self.candidates.items()},
        }


class CrosswordAgent:
    def __init__(
        self,
        oracle: Oracle,
        config: AgentConfig | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.oracle = oracle
        self.config = config or AgentConfig()
        self.on_event = on_event

    # ------------------------------------------------------------------------------
    def _emit(self, trace: list[dict], grid: Grid, **event) -> None:
        trace.append(event)
        if self.on_event:
            self.on_event(event, grid)

    @staticmethod
    def _context(puzzle: Puzzle) -> str:
        parts = [f"Puzzle: {puzzle.title or 'untitled'} ({puzzle.width}x{puzzle.height})."]
        if puzzle.notes:
            parts.append(f"Notes: {puzzle.notes}")
        if puzzle.difficulty:
            parts.append(f"Stated difficulty: {puzzle.difficulty}.")
        return " ".join(parts)

    @staticmethod
    def _uncommit(grid: Grid, slot: Slot, committed: set[str], puzzle: Puzzle) -> None:
        """Clear a committed slot, keeping letters that another committed slot also owns."""
        committed.discard(slot.id)
        for cell in slot.cells:
            owners = [sid for sid, _ in puzzle.cell_slots[cell] if sid != slot.id]
            if not any(o in committed for o in owners):
                grid.letters[cell] = None

    # ------------------------------------------------------------------------------
    def solve(self, puzzle: Puzzle) -> SolveResult:
        cfg = self.config
        t0 = time.time()
        committed = Grid(puzzle)  # only high-confidence letters live here
        committed_ids: set[str] = set()
        commit_conf: dict[str, float] = {}  # posterior at the moment each entry was committed
        table: CandidateTable = {}
        asked: dict[str, set[str]] = {s.id: set() for s in puzzle.slots}
        trace: list[dict] = []
        context = self._context(puzzle)
        posteriors: Posteriors = {}
        rounds = 0

        # ---- propose / reason / commit rounds --------------------------------------
        for rnd in range(1, cfg.max_rounds + 1):
            rounds = rnd
            queries = []
            for s in puzzle.slots:
                if s.id in committed_ids:
                    continue
                pat = committed.pattern(s)
                if pat not in asked[s.id]:
                    asked[s.id].add(pat)
                    queries.append(ClueQuery(s.id, s.clue, s.length, pat))
            added = 0
            if queries:
                self._emit(
                    trace, committed, type="stage", phase="propose", round=rnd,
                    queries=len(queries), text=f"Generating candidates for {len(queries)} clues.",
                )
                new = self.oracle.propose(queries, k=cfg.k, context=context, source=f"r{rnd}")
                added = merge_candidates(table, new)
            self._emit(trace, committed, type="propose", round=rnd, queries=len(queries), new=added)

            solver_cfg = SolverConfig(**{**cfg.solver.__dict__})
            solver_cfg.commit_threshold = max(
                cfg.min_threshold, cfg.solver.commit_threshold - cfg.threshold_decay * (rnd - 1)
            )
            self._emit(trace, committed, type="stage", phase="propagate", round=rnd,
                       text="Comparing candidate answers at their crossings.")
            posteriors = propagate(puzzle, committed, table, solver_cfg)
            commits = choose_commits(puzzle, committed, posteriors, solver_cfg)
            for slot, word, p in commits:
                committed.set_word(slot, word)
                committed_ids.add(slot.id)
                commit_conf[slot.id] = p
                self._emit(trace, committed, type="commit", round=rnd, slot=slot.id, word=word, p=round(p, 3))
            self._emit(
                trace, committed, type="round", round=rnd, commits=len(commits),
                filled=round(committed.filled_fraction(), 3), threshold=solver_cfg.commit_threshold,
            )
            if committed.is_complete():
                break
            if not commits and not queries:
                break  # nothing new to ask, nothing new to commit

        # ---- complete with search ---------------------------------------------------
        self._emit(trace, committed, type="stage", phase="propagate",
                   text="Updating candidates using the committed letters.")
        posteriors = propagate(puzzle, committed, table, cfg.solver)
        self._emit(trace, committed, type="stage", phase="search",
                   text="Searching for a compatible combination of answers.")
        grid, score, nodes = backtrack_fill(puzzle, committed, posteriors, cfg.solver)
        self._emit(trace, grid, type="backtrack", nodes=nodes, score=round(score, 2),
                   filled=round(grid.filled_fraction(), 3))

        # ---- judge & repair ---------------------------------------------------------
        for cycle in range(cfg.repair_cycles):
            conf = slot_confidences(puzzle, grid, posteriors)
            suspects = [
                JudgeQuery(s.id, s.clue, grid.word(s), committed.pattern(s))  # type: ignore[arg-type]
                for s in puzzle.slots
                if grid.word(s) is not None and conf[s.id] < cfg.judge_below
            ]
            if not suspects:
                break
            self._emit(
                trace, grid, type="stage", phase="judge", cycle=cycle + 1,
                queries=len(suspects), text=f"Checking {len(suspects)} uncertain answers against their clues.",
            )
            verdicts = self.oracle.judge(suspects, context=context)
            rejected = 0
            for q in suspects:
                v = verdicts.get(q.slot_id)
                if v is None or v.accept:
                    continue
                rejected += 1
                slot = puzzle.slot_by_id[q.slot_id]
                bucket = table.setdefault(slot.id, {})
                bucket.pop(q.answer, None)  # a rejected word is out; leaving the slot open beats it
                if v.alternative and len(v.alternative) == slot.length:
                    bucket[v.alternative] = max(bucket.get(v.alternative, 0.0), v.confidence)
                if slot.id in committed_ids:
                    self._uncommit(committed, slot, committed_ids, puzzle)
                    commit_conf.pop(slot.id, None)
                # a wrong entry usually rode in on a wrong crossing letter: free the weak crossings
                # too, otherwise the re-ask is stuck with the same corrupt pattern
                loosened = []
                for _i, other, _j in puzzle.crossings(slot):
                    if other.id in committed_ids and commit_conf.get(other.id, 1.0) < cfg.loosen_crossings_below:
                        self._uncommit(committed, other, committed_ids, puzzle)
                        commit_conf.pop(other.id, None)
                        loosened.append(other.id)
                self._emit(trace, grid, type="reject", cycle=cycle + 1, slot=slot.id,
                           word=q.answer, alternative=v.alternative, loosened=loosened)
            self._emit(trace, grid, type="judge", cycle=cycle + 1, checked=len(suspects), rejected=rejected)
            if not rejected:
                break
            # re-ask for the slots whose committed pattern changed, then re-solve
            queries = []
            for s in puzzle.slots:
                if s.id in committed_ids:
                    continue
                pat = committed.pattern(s)
                if pat not in asked[s.id]:
                    asked[s.id].add(pat)
                    queries.append(ClueQuery(s.id, s.clue, s.length, pat))
            if queries:
                self._emit(
                    trace, committed, type="stage", phase="propose", cycle=cycle + 1,
                    queries=len(queries), text=f"Generating replacement candidates for {len(queries)} clues.",
                )
                new = self.oracle.propose(queries, k=cfg.k, context=context, source=f"repair{cycle + 1}")
                merge_candidates(table, new)
            self._emit(trace, committed, type="stage", phase="propagate", cycle=cycle + 1,
                       text="Comparing replacement candidates at their crossings.")
            posteriors = propagate(puzzle, committed, table, cfg.solver)
            for slot, word, p in choose_commits(puzzle, committed, posteriors, cfg.solver):
                committed.set_word(slot, word)
                committed_ids.add(slot.id)
                commit_conf[slot.id] = p
                self._emit(trace, committed, type="commit", cycle=cycle + 1,
                           slot=slot.id, word=word, p=round(p, 3))
            self._emit(trace, committed, type="stage", phase="propagate", cycle=cycle + 1,
                       text="Updating candidates using the revised letters.")
            posteriors = propagate(puzzle, committed, table, cfg.solver)
            self._emit(trace, committed, type="stage", phase="search", cycle=cycle + 1,
                       text="Searching for a compatible revision of the grid.")
            grid, score, nodes = backtrack_fill(puzzle, committed, posteriors, cfg.solver)
            self._emit(trace, grid, type="backtrack", cycle=cycle + 1, nodes=nodes,
                       score=round(score, 2), filled=round(grid.filled_fraction(), 3))

        # ---- force fill -------------------------------------------------------------
        forced: set[str] = set()
        if cfg.force_fill and not grid.is_complete():
            open_slots = [s for s in puzzle.slots if grid.word(s) is None]
            queries = [ClueQuery(s.id, s.clue, s.length, grid.pattern(s)) for s in open_slots]
            self._emit(
                trace, grid, type="stage", phase="force_fill", queries=len(queries),
                text=f"Requesting best guesses for {len(queries)} unfinished entries.",
            )
            new = self.oracle.propose(queries, k=3, context=context, source="force")
            merge_candidates(table, new)
            # most constrained first so guesses cascade sensibly
            open_slots.sort(key=lambda s: -sum(1 for c in s.cells if grid.letters[c]))
            for s in open_slots:
                if grid.word(s) is not None:
                    continue
                options = sorted(table.get(s.id, {}).items(), key=lambda kv: -kv[1])
                for word, _ in options:
                    if grid.fits(s, word):
                        grid.set_word(s, word)
                        forced.add(s.id)
                        self._emit(trace, grid, type="force_fill", slot=s.id, word=word)
                        break

        # Confidence per entry: the posterior at commit time for committed entries (the
        # grid was less constrained then, so this is the honest belief); the posterior over
        # the committed grid for entries chosen by search; a low floor for forced guesses.
        confidences = slot_confidences(puzzle, grid, posteriors)
        for sid in confidences:
            slot = puzzle.slot_by_id[sid]
            w = grid.word(slot)
            if w is None:
                confidences[sid] = 0.0
            elif sid in commit_conf:
                confidences[sid] = commit_conf[sid]
            elif sid in forced or confidences[sid] == 0.0:
                raw = table.get(sid, {}).get(w, 0.05)
                confidences[sid] = min(0.3, 0.3 * raw + posteriors.get(sid, {}).get(OTHER, 0.0) * 0.1)
        elapsed = time.time() - t0
        self._emit(trace, grid, type="done", elapsed_s=round(elapsed, 2),
                   filled=round(grid.filled_fraction(), 3))
        usage = getattr(getattr(self.oracle, "client", None), "usage", None)
        model = getattr(getattr(getattr(self.oracle, "client", None), "config", None), "model", None)
        return SolveResult(
            puzzle=puzzle,
            grid=grid,
            committed=committed,
            confidences=confidences,
            posteriors=posteriors,
            candidates=table,
            trace=trace,
            rounds=rounds,
            elapsed_s=elapsed,
            oracle_stats=self.oracle.stats.to_dict(),
            usage=usage.to_dict() if usage else None,
            model=model,
        )
