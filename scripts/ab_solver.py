"""Deterministic A/B of solver settings using the simulated oracle on real held-out grids.

    .venv/bin/python scripts/ab_solver.py --n 40 --recall 0.6 --top1 0.55
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from crossword_agent.agent import AgentConfig, CrosswordAgent
from crossword_agent.evaluate import load_puzzle_set, score
from crossword_agent.oracle import SimulatedOracle
from crossword_agent.solver import SolverConfig


def run(items, recall, top1, pattern_boost, seed, **solver_kw):
    accs, words = [], []
    for i, (_, p) in enumerate(items):
        oracle = SimulatedOracle(p, recall=recall, top1=top1, pattern_boost=pattern_boost, seed=seed + i)
        cfg = AgentConfig(solver=SolverConfig(**solver_kw))
        m = score(CrosswordAgent(oracle, cfg).solve(p))
        accs.append(m.letter_accuracy)
        words.append(m.word_accuracy)
    return statistics.fmean(accs), statistics.fmean(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus/test.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--recall", type=float, default=0.6)
    ap.add_argument("--top1", type=float, default=0.55)
    ap.add_argument("--pattern-boost", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = load_puzzle_set([args.corpus], limit=args.n)
    variants = {
        "ceiling 0.6 (old)": {"other_ceiling": 0.6},
        "ceiling 0.97": {"other_ceiling": 0.97},
        "ceiling 0.97, threshold 0.85": {"other_ceiling": 0.97, "commit_threshold": 0.85},
        "ceiling 0.97, damping 0.5": {"other_ceiling": 0.97, "damping": 0.5},
        "ceiling 0.97, damping 1.0": {"other_ceiling": 0.97, "damping": 1.0},
    }
    print(f"{args.n} grids, recall {args.recall}, top1 {args.top1}, pattern boost {args.pattern_boost}")
    for name, kw in variants.items():
        la, wa = run(items, args.recall, args.top1, args.pattern_boost, args.seed, **kw)
        print(f"  {name:<32} letters {la:.1%}  words {wa:.1%}")


if __name__ == "__main__":
    main()
