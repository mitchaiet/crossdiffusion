"""Clue oracles: things that turn (clue, length, pattern) into ranked candidate answers.

`LLMOracle` asks the language model (batched, concurrent).
`SimulatedOracle` fakes an answerer of tunable quality from the known solution, so the
constraint solver can be tested and stress-tested independently of any model.
"""

from __future__ import annotations

import json
import logging
import random
import re
import string
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from .grid import Puzzle
from .llm import LLMError

log = logging.getLogger(__name__)
_NON_ALPHA = re.compile(r"[^A-Z]")


@dataclass(frozen=True)
class ClueQuery:
    slot_id: str
    clue: str
    length: int
    pattern: str  # '_' for unknown letters, e.g. "_A__E"


@dataclass(frozen=True)
class JudgeQuery:
    slot_id: str
    clue: str
    answer: str  # the word currently in the grid
    pattern: str  # letters we are confident about ('_' elsewhere)


@dataclass
class Candidate:
    answer: str
    confidence: float
    source: str = "llm"


@dataclass
class Judgement:
    accept: bool
    confidence: float
    alternative: str | None = None


@dataclass
class OracleStats:
    """Constraint-compliance bookkeeping, useful for evaluating the answerer itself."""

    queries: int = 0
    raw_candidates: int = 0
    invalid_candidates: int = 0  # wrong length / pattern mismatch / not letters
    empty_responses: int = 0
    failed_batches: int = 0  # LLM batches that failed and were split / dropped

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Oracle(Protocol):
    stats: OracleStats

    def propose(self, queries: list[ClueQuery], *, k: int, context: str) -> dict[str, list[Candidate]]: ...

    def judge(self, queries: list[JudgeQuery], *, context: str) -> dict[str, Judgement]: ...


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def normalize_answer(text: str) -> str:
    """Uppercase, fold accents, strip everything that is not A-Z (answers ignore spaces/punct)."""
    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return _NON_ALPHA.sub("", folded.upper())


def matches_pattern(word: str, pattern: str) -> bool:
    if len(word) != len(pattern):
        return False
    return all(p == "_" or p == w for p, w in zip(pattern, word, strict=True))


def _clean_candidates(
    raw: list, length: int, pattern: str, stats: OracleStats, source: str
) -> list[Candidate]:
    out: dict[str, Candidate] = {}
    for item in raw or []:
        stats.raw_candidates += 1
        if isinstance(item, str):
            answer, conf = item, 0.5
        elif isinstance(item, dict):
            answer = item.get("answer", "")
            try:
                conf = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
        else:
            stats.invalid_candidates += 1
            continue
        word = normalize_answer(answer)
        if len(word) != length or not matches_pattern(word, pattern):
            stats.invalid_candidates += 1
            continue
        conf = min(1.0, max(0.01, conf))
        if word not in out or out[word].confidence < conf:
            out[word] = Candidate(word, conf, source)
    return sorted(out.values(), key=lambda c: -c.confidence)


# --------------------------------------------------------------------------------------
# LLM oracle
# --------------------------------------------------------------------------------------
PROPOSE_SYSTEM = """You are a world-class solver of American-style crossword puzzles.

You receive a batch of clues. Each has an exact answer length and a letter pattern
('_' = unknown letter, a letter = that letter is fixed by a crossing word).

Rules for answers:
- Uppercase letters A-Z only. Drop spaces, hyphens and punctuation: "à la carte" -> ALACARTE, "J.F.K." -> JFK.
- The answer MUST have exactly the stated length and MUST match every fixed letter of the pattern.
- Crossword conventions: "Abbr."/"briefly"/"for short" -> abbreviation; a plural clue -> plural answer;
  match verb tense; "?" signals wordplay or a pun; "___" is a fill-in-the-blank; a clue in another
  language wants an answer in that language; "e.g."/"for one" wants a category member; the puzzle
  title or theme may hint at a gimmick.
- Rank up to K distinct candidates per clue, best first. `confidence` is your calibrated probability
  that the candidate is exactly right; the confidences for one clue should sum to at most 1.
- When unsure, still offer your best guesses with low confidence rather than nothing. Prefer real
  words/phrases that fit the pattern over invented strings.
- Think efficiently: a few seconds of consideration per clue, then answer. Do not deliberate at length.
- A `draft` answer, when present, came from a separate grid model. Treat it as a fallible
  suggestion: check it against the clue and crossing pattern, and replace it when needed.

Respond with JSON only:
{"answers": [{"id": "<clue id>", "candidates": [{"answer": "WORD", "confidence": 0.8}, ...]}, ...]}
Include every clue id you were given."""

JUDGE_SYSTEM = """You are a meticulous crossword editor checking a solver's filled grid.

For each item you get a clue, the answer currently written in the grid, and the letters the solver is
confident about ('_' = uncertain). Decide whether the written answer is a correct, fair answer to the
clue. If it is wrong, propose the single best alternative that fits the confident letters (uppercase,
same length), or null if you have none.

Respond with JSON only:
{"judgements": [{"id": "<id>", "accept": true, "confidence": 0.9, "alternative": null}, ...]}
Include every id you were given."""


class LLMOracle:
    def __init__(self, client, *, batch_size: int = 12, concurrency: int = 4,
                 draft_words: dict[str, str] | None = None) -> None:
        self.client = client
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.draft_words = draft_words or {}
        self.stats = OracleStats()

    # -- propose --------------------------------------------------------------------
    def _propose_batch(self, batch: list[ClueQuery], k: int, context: str, source: str):
        payload = [
            {"id": q.slot_id, "clue": q.clue, "length": q.length, "pattern": q.pattern,
             **({"draft": self.draft_words[q.slot_id]} if q.slot_id in self.draft_words else {})}
            for q in batch
        ]
        user = (
            f"{context}\n\nK = {k}\n\nClues:\n{json.dumps(payload, ensure_ascii=False, indent=0)}"
        )
        data = self.client.complete_json(PROPOSE_SYSTEM, user)
        by_id = {q.slot_id: q for q in batch}
        result: dict[str, list[Candidate]] = {q.slot_id: [] for q in batch}
        answers = data.get("answers", []) if isinstance(data, dict) else []
        for entry in answers:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("id", "")).upper().replace(" ", "")
            q = by_id.get(sid)
            if q is None:
                continue
            result[sid] = _clean_candidates(
                entry.get("candidates", []), q.length, q.pattern, self.stats, source
            )
        for cands in result.values():
            if not cands:
                self.stats.empty_responses += 1
        return result

    def _propose_resilient(self, batch: list[ClueQuery], k: int, context: str, source: str) -> dict[str, list[Candidate]]:
        """A batch that fails (runaway reasoning, malformed JSON) is split in half and retried;
        a single clue that still fails yields no candidates rather than aborting the solve."""
        try:
            return self._propose_batch(batch, k, context, source)
        except LLMError as e:
            self.stats.failed_batches += 1
            if len(batch) == 1:
                log.warning("giving up on clue %s: %s", batch[0].slot_id, e)
                self.stats.empty_responses += 1
                return {batch[0].slot_id: []}
            mid = len(batch) // 2
            out = self._propose_resilient(batch[:mid], k, context, source)
            out.update(self._propose_resilient(batch[mid:], k, context, source))
            return out

    def propose(self, queries: list[ClueQuery], *, k: int, context: str, source: str = "llm") -> dict[str, list[Candidate]]:
        self.stats.queries += len(queries)
        batches = [queries[i : i + self.batch_size] for i in range(0, len(queries), self.batch_size)]
        merged: dict[str, list[Candidate]] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as ex:
            for part in ex.map(lambda b: self._propose_resilient(b, k, context, source), batches):
                merged.update(part)
        return merged

    # -- judge ----------------------------------------------------------------------
    def _judge_batch(self, batch: list[JudgeQuery], context: str) -> dict[str, Judgement]:
        payload = [
            {"id": q.slot_id, "clue": q.clue, "written": q.answer, "confident_letters": q.pattern}
            for q in batch
        ]
        user = f"{context}\n\nItems:\n{json.dumps(payload, ensure_ascii=False, indent=0)}"
        data = self.client.complete_json(JUDGE_SYSTEM, user)
        by_id = {q.slot_id: q for q in batch}
        out: dict[str, Judgement] = {}
        for entry in data.get("judgements", []) if isinstance(data, dict) else []:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("id", "")).upper().replace(" ", "")
            q = by_id.get(sid)
            if q is None:
                continue
            alt = entry.get("alternative")
            alt_word = normalize_answer(alt) if alt else None
            if alt_word and not matches_pattern(alt_word, q.pattern):
                alt_word = None
            try:
                conf = float(entry.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            out[sid] = Judgement(bool(entry.get("accept", True)), min(1.0, max(0.0, conf)), alt_word)
        return out

    def _judge_resilient(self, batch: list[JudgeQuery], context: str) -> dict[str, Judgement]:
        try:
            return self._judge_batch(batch, context)
        except LLMError as e:
            self.stats.failed_batches += 1
            if len(batch) == 1:
                log.warning("judge gave up on %s: %s", batch[0].slot_id, e)
                return {}
            mid = len(batch) // 2
            out = self._judge_resilient(batch[:mid], context)
            out.update(self._judge_resilient(batch[mid:], context))
            return out

    def judge(self, queries: list[JudgeQuery], *, context: str) -> dict[str, Judgement]:
        batches = [queries[i : i + self.batch_size] for i in range(0, len(queries), self.batch_size)]
        merged: dict[str, Judgement] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as ex:
            for part in ex.map(lambda b: self._judge_resilient(b, context), batches):
                merged.update(part)
        return merged


# --------------------------------------------------------------------------------------
# Simulated oracle (for tests and solver stress tests)
# --------------------------------------------------------------------------------------
class SimulatedOracle:
    """Fakes a clue answerer of controllable quality using the known solution.

    recall      probability the true answer appears among the candidates (no pattern known)
    top1        probability the true answer is ranked first, given it appears
    pattern_boost  how much a partially known pattern raises recall (LLMs get much better
                   once crossings pin letters down; this mimics that)
    judge_accuracy probability the judge classifies a written answer correctly
    """

    def __init__(
        self,
        puzzle: Puzzle,
        *,
        recall: float = 0.7,
        top1: float = 0.6,
        pattern_boost: float = 0.8,
        judge_accuracy: float = 0.9,
        seed: int = 0,
    ) -> None:
        if puzzle.solution is None:
            raise ValueError("SimulatedOracle needs a puzzle with a solution")
        self.puzzle = puzzle
        self.recall = recall
        self.top1 = top1
        self.pattern_boost = pattern_boost
        self.judge_accuracy = judge_accuracy
        self.rng = random.Random(seed)
        self.stats = OracleStats()

    def _distractor(self, truth: str, pattern: str) -> str:
        """A plausible-looking wrong answer that still fits the pattern."""
        letters = list(truth)
        free = [i for i, p in enumerate(pattern) if p == "_"]
        if not free:
            return truth
        n_changes = min(len(free), self.rng.choice([1, 1, 2, 2, 3]))
        for i in self.rng.sample(free, n_changes):
            letters[i] = self.rng.choice(string.ascii_uppercase.replace(letters[i], ""))
        return "".join(letters)

    def propose(self, queries: list[ClueQuery], *, k: int, context: str = "", source: str = "sim") -> dict[str, list[Candidate]]:
        self.stats.queries += len(queries)
        out: dict[str, list[Candidate]] = {}
        for q in queries:
            truth = self.puzzle.solution_word(self.puzzle.slot_by_id[q.slot_id]) or ""
            known = sum(1 for p in q.pattern if p != "_") / max(1, q.length)
            eff_recall = self.recall + (1 - self.recall) * known * self.pattern_boost
            cands: list[Candidate] = []
            has_truth = self.rng.random() < eff_recall
            n_distract = self.rng.randint(1, max(1, k - 1))
            seen = {truth}
            for _ in range(n_distract):
                d = self._distractor(truth, q.pattern)
                if d not in seen:
                    seen.add(d)
                    cands.append(Candidate(d, round(self.rng.uniform(0.1, 0.6), 3), source))
            # pattern-fitting junk at near-zero confidence, as LLMs produce when they have no idea
            for _ in range(self.rng.randint(0, 2)):
                j = self._distractor(truth, q.pattern)
                if j not in seen:
                    seen.add(j)
                    cands.append(Candidate(j, round(self.rng.uniform(0.01, 0.08), 3), source))
            if has_truth:
                is_top = self.rng.random() < (self.top1 + (1 - self.top1) * known)
                conf = self.rng.uniform(0.6, 0.95) if is_top else self.rng.uniform(0.15, 0.5)
                cands.append(Candidate(truth, round(conf, 3), source))
                if is_top:
                    for c in cands:
                        if c.answer != truth:
                            c.confidence = min(c.confidence, conf - 0.05)
            cands.sort(key=lambda c: -c.confidence)
            self.stats.raw_candidates += len(cands)
            out[q.slot_id] = cands[:k]
        return out

    def judge(self, queries: list[JudgeQuery], *, context: str = "") -> dict[str, Judgement]:
        out: dict[str, Judgement] = {}
        for q in queries:
            truth = self.puzzle.solution_word(self.puzzle.slot_by_id[q.slot_id]) or ""
            correct = q.answer == truth
            right_call = self.rng.random() < self.judge_accuracy
            accept = correct if right_call else not correct
            alt = None
            if not accept and right_call and matches_pattern(truth, q.pattern):
                alt = truth
            out[q.slot_id] = Judgement(accept, round(self.rng.uniform(0.6, 0.95), 3), alt)
        return out
