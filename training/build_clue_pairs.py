"""Knowledge injection data: clue -> answer pairs as one-row puzzles.

A single-row grid with one across entry goes through exactly the same canvas format as a real
grid, so the model practises answering clues from scratch (the skill it is weakest at) without
any new code path.

    python -m training.build_clue_pairs --clues data/raw/xd-clues/xd/clues.tsv --n 200000 --out data/corpus/clue_pairs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

csv.field_size_limit(1 << 26)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clues", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=15)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="puzzle JSONL files whose (answer, clue) pairs must not appear (test sets)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    banned: set[tuple[str, str]] = set()
    for f in args.exclude:
        for line in f.open():
            d = json.loads(line)
            from crossword_agent.grid import Puzzle

            pz = Puzzle.from_rows(d["grid"], d.get("clues", {}))
            for s in pz.slots:
                banned.add((pz.solution_word(s), s.clue.strip().lower()))
    print(f"excluding {len(banned)} test (answer, clue) pairs")

    rng = random.Random(args.seed)
    seen: set[tuple[str, str]] = set()
    rows = []
    with args.clues.open(encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            ans, clue = row[2].strip().upper(), row[3].strip()
            # ASCII letters only: the canvas has 26 letter tokens (str.isalpha accepts É, Ñ, ...)
            if not re.fullmatch(r"[A-Z]+", ans) or not (args.min_len <= len(ans) <= args.max_len) or len(clue) < 2:
                continue
            key = (ans, clue.lower())
            if key in seen or key in banned:
                continue
            seen.add(key)
            rows.append((ans, clue))
    print(f"{len(rows)} distinct usable (answer, clue) pairs")
    rng.shuffle(rows)
    rows = rows[: args.n]
    with args.out.open("w") as out:
        for ans, clue in rows:
            out.write(json.dumps({"title": "", "grid": [ans], "clues": {"across": {"1": clue}, "down": {}}, "source": "clue-pair"}) + "\n")
    print(f"wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
