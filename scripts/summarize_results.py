"""Render every results/*.json run into a Markdown table and splice it into docs/EVALUATION.md.

    .venv/bin/python scripts/summarize_results.py            # prints the table
    .venv/bin/python scripts/summarize_results.py --write    # also updates docs/EVALUATION.md (## Results)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(results_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if "summary" in d and d["summary"]:
            runs.append(d)
    return runs


def puzzle_set_label(config: dict) -> str:
    puzzles = config.get("puzzles", [])
    if not puzzles:
        return "?"
    first = str(puzzles[0])
    if "test_nyt_sat" in first:
        return f"NYT Saturday 15x15 (held out, n={len(puzzles)})"
    if "test_real_mixed" in first:
        return f"mixed publishers 15x15 (held out, n={len(puzzles)})"
    if first.startswith("puzzles/") or "/puzzles/" in first:
        return f"hand-clued benchmark (n={len(puzzles)})"
    if first.startswith("test#"):
        return f"random held-out 15x15 (n={len(puzzles)})"
    return f"{Path(first).stem} (n={len(puzzles)})"


def solver_label(config: dict) -> str:
    mode = config.get("mode", "?")
    if mode == "hybrid":
        hm = (config.get("diffusion") or {}).get("mode", "inject")
        return "hybrid (fallback)" if hm == "fallback" else "hybrid"
    if mode != "agent":
        return mode
    notes = []
    if config.get("rounds") == 1:
        notes.append("1 round")
    if config.get("repair") == 0:
        notes.append("no judge")
    return "agent" + (f" ({', '.join(notes)})" if notes else "")


def render(runs: list[dict]) -> str:
    rows = ["| puzzle set | solver | model | solved | letter acc | word acc | fill | precision | ECE | s/puzzle | $/puzzle |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"oneshot": 0, "top1": 1, "agent": 2, "hybrid": 3, "diffusion": 4, "simulate": 5}
    runs = sorted(runs, key=lambda r: (puzzle_set_label(r["config"]), order.get(r["config"].get("mode"), 9),
                                       -(r["config"].get("repair", 1) == 0) - (r["config"].get("rounds", 4) == 1) * 2, r["name"]))
    for r in runs:
        s, c = r["summary"], r["config"]
        cost = f"${s['cost_usd'] / s['puzzles']:.3f}" if s.get("cost_usd") is not None else "-"
        ece = s["calibration"].get("ece")
        model = str(c.get("model", "")).split("/")[-1]
        if c.get("mode") in ("diffusion", "hybrid") and "__" in r["name"]:
            model = r["name"].split("__", 1)[1]  # run names carry the checkpoint, e.g. gemma1b-20k
        rows.append(
            f"| {puzzle_set_label(c)} | {solver_label(c)} | {model} | "
            f"{s['solved']}/{s['puzzles']} | {s['letter_accuracy']:.1%} | {s['word_accuracy']:.1%} | "
            f"{s['fill_rate']:.0%} | {s['letter_precision']:.1%} | {ece if ece is not None else '-'} | "
            f"{s['mean_elapsed_s']:.0f} | {cost} |"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--doc", type=Path, default=Path("docs/EVALUATION.md"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    table = render(load_runs(args.results))
    print(table)
    if args.write:
        text = args.doc.read_text()
        head, sep, _ = text.partition("## Results")
        if not sep:
            head, sep = text + "\n", "## Results"
        body = ("\n\nAll numbers from `results/*.json` (temperature 0, responses cached). "
                "Letter accuracy counts blanks as wrong; precision is correct / filled. ECE is the expected "
                "calibration error of the per-entry confidence (one-shot reports a flat 0.5, hence its ECE).\n\n"
                + table + "\n")
        args.doc.write_text(head + sep + body)
        print(f"\nupdated {args.doc}")


if __name__ == "__main__":
    main()
