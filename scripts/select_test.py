"""Pick a held-out test subset from the corpus splits by publisher / weekday / size.

    .venv/bin/python scripts/select_test.py --pub nyt --weekday Sat --size 15x15 --n 30 \
        --out data/corpus/test_nyt_sat.jsonl

Reads data/corpus/test.jsonl (and val.jsonl if --include-val) so nothing here was trained on.
The xd file name encodes publisher and date: .../nyt2019-03-16.xd
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

_NAME = re.compile(r"([a-z]+)(\d{4}-\d{2}-\d{2})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--include-val", action="store_true")
    ap.add_argument("--pub", default=None, help="publisher id, e.g. nyt, lat, wsj")
    ap.add_argument("--weekday", default=None, help="Mon..Sun")
    ap.add_argument("--size", default=None, help="e.g. 15x15")
    ap.add_argument("--year-min", type=int, default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = [args.corpus / "test.jsonl"] + ([args.corpus / "val.jsonl"] if args.include_val else [])
    picked = []
    for f in files:
        for line in f.open():
            d = json.loads(line)
            m = _NAME.search(Path(d.get("source", "")).stem)
            if not m:
                continue
            pub, date = m.group(1), dt.date.fromisoformat(m.group(2))
            if args.pub and pub != args.pub:
                continue
            if args.weekday and date.strftime("%a") != args.weekday:
                continue
            if args.year_min and date.year < args.year_min:
                continue
            size = f"{len(d['grid'][0])}x{len(d['grid'])}"
            if args.size and size != args.size:
                continue
            d["title"] = d.get("title") or f"{pub} {date}"
            picked.append(d)
            if len(picked) >= args.n:
                break
        if len(picked) >= args.n:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for d in picked:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} puzzles -> {args.out}")


if __name__ == "__main__":
    main()
