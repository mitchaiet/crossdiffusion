"""Evaluation: scoring a solve against the known solution, aggregation, calibration, reports."""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import SolveResult
from .grid import Puzzle
from .llm import estimate_cost_usd

log = logging.getLogger(__name__)
Solver = Callable[[Puzzle], SolveResult]


@dataclass
class PuzzleMetrics:
    file: str
    title: str
    size: str
    difficulty: str
    n_cells: int
    n_slots: int
    letters_correct: int
    letters_filled: int
    words_correct: int
    words_filled: int
    letter_accuracy: float  # correct / all white cells (blanks count as wrong)
    letter_precision: float  # correct / filled cells
    fill_rate: float
    word_accuracy: float
    solved: bool
    elapsed_s: float
    llm_latency_s: float  # summed model latency, live + cached (parallel batches overlap in wall time)
    rounds: int
    llm_calls: int
    cache_hits: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    oracle_stats: dict = field(default_factory=dict)
    slot_calibration: list[tuple[float, bool]] = field(default_factory=list)
    wrong_slots: list[dict] = field(default_factory=list)


def score(result: SolveResult, file: str = "") -> PuzzleMetrics:
    p = result.puzzle
    if p.solution is None:
        raise ValueError("cannot score a puzzle without a solution")
    cells = p.white_cells
    filled = [c for c in cells if result.grid.letters[c]]
    correct = [c for c in cells if result.grid.letters[c] == p.solution[c]]
    words_filled, words_correct, calib, wrong = 0, 0, [], []
    for s in p.slots:
        w = result.grid.word(s)
        truth = p.solution_word(s)
        if w is not None:
            words_filled += 1
        ok = w == truth
        if ok:
            words_correct += 1
        else:
            wrong.append({"slot": s.id, "clue": s.clue, "got": w, "want": truth})
        calib.append((float(result.confidences.get(s.id, 0.0)), ok))
    usage = result.usage or {}
    # cost and LLM time include cache replays, so a cached re-run reports what a live run costs
    pt = usage.get("prompt_tokens", 0) + usage.get("cached_prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0) + usage.get("cached_completion_tokens", 0)
    return PuzzleMetrics(
        file=file or p.source,
        title=p.title,
        size=f"{p.width}x{p.height}",
        difficulty=p.difficulty,
        n_cells=len(cells),
        n_slots=len(p.slots),
        letters_correct=len(correct),
        letters_filled=len(filled),
        words_correct=words_correct,
        words_filled=words_filled,
        letter_accuracy=len(correct) / len(cells),
        letter_precision=(len(correct) / len(filled)) if filled else 0.0,
        fill_rate=len(filled) / len(cells),
        word_accuracy=words_correct / len(p.slots),
        solved=len(correct) == len(cells),
        elapsed_s=result.elapsed_s,
        llm_latency_s=round(usage.get("latency_s", 0.0) + usage.get("cached_latency_s", 0.0), 2),
        rounds=result.rounds,
        llm_calls=usage.get("calls", 0) + usage.get("cache_hits", 0),
        cache_hits=usage.get("cache_hits", 0),
        prompt_tokens=pt,
        completion_tokens=ct,
        cost_usd=estimate_cost_usd(result.model, pt, ct) if result.model else None,
        oracle_stats=result.oracle_stats,
        slot_calibration=calib,
        wrong_slots=wrong,
    )


def calibration(pairs: Iterable[tuple[float, bool]], bins: int = 10) -> dict:
    """Expected calibration error over slot confidences (10 equal-width bins)."""
    pairs = list(pairs)
    if not pairs:
        return {"ece": None, "bins": []}
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, ok in pairs:
        i = min(bins - 1, int(conf * bins))
        buckets[i].append((conf, ok))
    ece, rows = 0.0, []
    for i, b in enumerate(buckets):
        if not b:
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, ok in b if ok) / len(b)
        ece += len(b) / len(pairs) * abs(acc - mean_conf)
        rows.append({"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": len(b),
                     "confidence": round(mean_conf, 3), "accuracy": round(acc, 3)})
    return {"ece": round(ece, 4), "bins": rows}


def aggregate(metrics: list[PuzzleMetrics]) -> dict:
    if not metrics:
        return {}
    def mean(key: str) -> float:
        return statistics.fmean(getattr(m, key) for m in metrics)
    costs = [m.cost_usd for m in metrics if m.cost_usd is not None]
    return {
        "puzzles": len(metrics),
        "solved": sum(1 for m in metrics if m.solved),
        "solve_rate": round(sum(1 for m in metrics if m.solved) / len(metrics), 4),
        "letter_accuracy": round(mean("letter_accuracy"), 4),
        "letter_precision": round(mean("letter_precision"), 4),
        "fill_rate": round(mean("fill_rate"), 4),
        "word_accuracy": round(mean("word_accuracy"), 4),
        "letters_correct": sum(m.letters_correct for m in metrics),
        "letters_total": sum(m.n_cells for m in metrics),
        "micro_letter_accuracy": round(
            sum(m.letters_correct for m in metrics) / sum(m.n_cells for m in metrics), 4
        ),
        "mean_elapsed_s": round(mean("elapsed_s"), 2),
        "total_elapsed_s": round(sum(m.elapsed_s for m in metrics), 1),
        "llm_calls": sum(m.llm_calls for m in metrics),
        "cache_hits": sum(m.cache_hits for m in metrics),
        "prompt_tokens": sum(m.prompt_tokens for m in metrics),
        "completion_tokens": sum(m.completion_tokens for m in metrics),
        "cost_usd": round(sum(costs), 4) if costs else None,
        "calibration": calibration(pair for m in metrics for pair in m.slot_calibration),
    }


PUZZLE_GLOBS = ("*.json", "*.puz")


def load_puzzle_set(paths: list[Path], limit: int | None = None) -> list[tuple[str, Puzzle]]:
    """Expand files / directories / JSONL corpora into (label, Puzzle) pairs."""
    items: list[tuple[str, Puzzle]] = []
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            for pat in PUZZLE_GLOBS:
                files.extend(sorted(p.rglob(pat)))
        else:
            files.append(p)
    for f in files:
        if f.suffix.lower() == ".jsonl":
            with f.open() as fh:
                for i, line in enumerate(fh):
                    d = json.loads(line)
                    pz = Puzzle.from_rows(d["grid"], d.get("clues", {}), title=d.get("title", ""),
                                          source=d.get("source", f"{f}#{i}"))
                    items.append((f"{f.stem}#{i}", pz))
                    if limit and len(items) >= limit:
                        return items
        else:
            items.append((str(f), Puzzle.load(f)))
        if limit and len(items) >= limit:
            break
    return items[:limit] if limit else items


def run_eval(
    items: list[tuple[str, Puzzle]], make_solver: Callable[[Puzzle], Solver], on_progress=None
) -> list[PuzzleMetrics]:
    out: list[PuzzleMetrics] = []
    for label, puzzle in items:
        solver = make_solver(puzzle)
        try:
            result = solver(puzzle)
        except Exception as exc:  # noqa: BLE001 - one failed puzzle must not sink the run
            log.error("solver failed on %s: %r", label, exc)
            from .grid import Grid

            empty = Grid(puzzle)
            result = SolveResult(puzzle=puzzle, grid=empty, committed=empty, confidences={},
                                 posteriors={}, candidates={}, trace=[{"type": "error", "error": repr(exc)}],
                                 rounds=0, elapsed_s=0.0, oracle_stats={})
        m = score(result, file=label)
        out.append(m)
        if on_progress:
            on_progress(m, result)
    return out


def write_report(name: str, config: dict, metrics: list[PuzzleMetrics], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(metrics)
    payload = {"name": name, "config": config, "summary": summary,
               "puzzles": [asdict(m) for m in metrics]}
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    md = [f"# Evaluation: {name}", "", "```", json.dumps(config, indent=2), "```", "",
          "## Summary", ""]
    for k, v in summary.items():
        if k != "calibration":
            md.append(f"- **{k}**: {v}")
    cal = summary.get("calibration", {})
    md += ["", f"Expected calibration error (slot confidence): **{cal.get('ece')}**", "",
           "| conf bin | n | mean conf | accuracy |", "|---|---|---|---|"]
    for row in cal.get("bins", []):
        md.append(f"| {row['bin']} | {row['n']} | {row['confidence']} | {row['accuracy']} |")
    md += ["", "## Per puzzle", "",
           "| file | size | diff | letters | words | solved | time (s) | LLM calls | cost |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in metrics:
        cost = f"${m.cost_usd:.4f}" if m.cost_usd is not None else "-"
        md.append(
            f"| {Path(m.file).name} | {m.size} | {m.difficulty or '-'} | "
            f"{m.letters_correct}/{m.n_cells} ({m.letter_accuracy:.0%}) | "
            f"{m.words_correct}/{m.n_slots} | {'yes' if m.solved else 'no'} | "
            f"{m.elapsed_s:.1f} | {m.llm_calls} | {cost} |"
        )
    md_path = out_dir / f"{name}.md"
    md_path.write_text("\n".join(md) + "\n")
    return json_path, md_path
