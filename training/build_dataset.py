"""Build the CrossDiffusion training corpus.

Sources
  real       .xd puzzles (published grids, clues, solutions)
  synthetic  real black-square layouts (from .xd) re-filled with words from the xd clue
             corpus, each entry paired with a randomly chosen real clue for that answer

Output: JSONL files of puzzles in the repo's JSON puzzle format (grid rows + clues),
split into train / val / test so the test grids are never seen in training.

    python -m training.build_dataset \
        --xd data/raw/xd-puzzles --clues data/raw/xd-clues/xd/clues.tsv \
        --out data/corpus --synthetic 20000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from construct import Filler  # noqa: E402  (scripts/construct.py)

from crossword_agent.grid import Puzzle  # noqa: E402
from training.xd_format import load_xd_dir  # noqa: E402

csv.field_size_limit(1 << 26)


def load_clue_bank(path: Path, min_uses: int = 3, max_len: int = 15) -> dict[str, list[str]]:
    """answer -> list of distinct clues, for answers used at least `min_uses` times."""
    bank: dict[str, set[str]] = defaultdict(set)
    uses: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            ans, clue = row[2].strip().upper(), row[3].strip()
            if not ans.isalpha() or not (3 <= len(ans) <= max_len) or not clue:
                continue
            uses[ans] += 1
            if len(bank[ans]) < 40:
                bank[ans].add(clue)
    return {a: sorted(c) for a, c in bank.items() if uses[a] >= min_uses}


def layouts_from(puzzles: list[Puzzle]) -> list[list[str]]:
    seen, out = set(), []
    for p in puzzles:
        rows = ["".join("#" if (r, c) in p.blocks else "." for c in range(p.width)) for r in range(p.height)]
        key = "\n".join(rows)
        if key not in seen:
            seen.add(key)
            out.append(rows)
    return out


def synthesize(layouts: list[list[str]], bank: dict[str, list[str]], n: int, seed: int,
               node_limit: int = 20_000, max_cells: int = 13 * 13, progress_every: int = 100) -> list[dict]:
    rng = random.Random(seed)
    words = list(bank)
    layouts = [lay for lay in layouts if len(lay) * len(lay[0]) <= max_cells] or layouts
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 6:
        attempts += 1
        if attempts % progress_every == 0:
            print(f"  synth: {len(out)}/{n} after {attempts} attempts", flush=True)
        rows = rng.choice(layouts)
        try:
            filler = Filler(rows, words, seed=rng.randrange(1 << 30), node_limit=node_limit)
        except SystemExit:
            continue
        res = filler.solve()
        if not res:
            continue
        letters = res["letters"]
        grid = ["".join("#" if ch == "#" else letters[(r, c)] for c, ch in enumerate(row)) for r, row in enumerate(rows)]
        clues = {"across": {}, "down": {}}
        for slot in filler.slots:
            word = res["assigned"][slot.id]
            clues["across" if slot.direction == "A" else "down"][str(slot.number)] = rng.choice(bank[word])
        out.append({"title": f"Synthetic {len(out) + 1}", "grid": grid, "clues": clues, "source": "synthetic"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xd", type=Path, required=True)
    ap.add_argument("--clues", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--synthetic", type=int, default=0, help="synthetic grids to add (written to train_synth.jsonl)")
    ap.add_argument("--synth-max-cells", type=int, default=13 * 13, help="only refill layouts up to this many cells")
    ap.add_argument("--synth-node-limit", type=int, default=20_000)
    ap.add_argument("--test", type=int, default=300, help="held-out real puzzles")
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--max-size", type=int, default=15, help="skip grids wider/taller than this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    real = [p for p in load_xd_dir(args.xd) if p.width <= args.max_size and p.height <= args.max_size]
    print(f"real puzzles usable: {len(real)}", flush=True)
    rng.shuffle(real)
    test, val, train_real = real[: args.test], real[args.test : args.test + args.val], real[args.test + args.val :]

    args.out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, items: list) -> None:
        path = args.out / f"{name}.jsonl"
        with path.open("w") as f:
            for it in items:
                d = it if isinstance(it, dict) else {**it.to_dict(), "source": it.source}
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"wrote {len(items):>7} -> {path}", flush=True)

    # real splits first: the slow synthetic stage must never hold these up
    dump("train", train_real)
    dump("val", val)
    dump("test", test)
    if not args.synthetic:
        return

    bank = load_clue_bank(args.clues)
    print(f"clue bank answers: {len(bank)}", flush=True)
    layouts = layouts_from(train_real)
    print(f"distinct layouts for synthesis: {len(layouts)}", flush=True)
    synth = synthesize(layouts, bank, args.synthetic, args.seed, node_limit=args.synth_node_limit,
                       max_cells=args.synth_max_cells)
    print(f"synthetic puzzles: {len(synth)}")
    dump("train_synth", synth)


if __name__ == "__main__":
    main()
