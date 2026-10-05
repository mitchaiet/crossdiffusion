"""Tiny HTTP server exposing the CrossDiffusion filler to the agent on another machine.

    python -m training.serve --base google/gemma-3-4b-it --adapter runs/gemma4b-r1/adapter --port 8009

POST /fill  {"grid": ["#FEUD", "C..R.", ...], "clues": {...}, "title": "...", "steps": 12}
            '#' black, letter = known, '.' unknown
       ->   {"grid": [...], "confidence": {"r,c": p}, "history": [...], "elapsed_s": s}
GET  /      health
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from crossword_agent.grid import Puzzle
from training.sample import DiffusionFiller

FILLER: DiffusionFiller | None = None
DEFAULT_STEPS = 12


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._json(200, {"ok": True, "model": getattr(FILLER, "name", "crossdiffusion")})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/fill":
            self._json(404, {"error": "unknown path"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(n))
            t0 = time.time()
            rows = req["grid"]
            layout = ["".join("#" if ch == "#" else "." for ch in row) for row in rows]
            puzzle = Puzzle.from_rows(layout, req.get("clues", {}), title=req.get("title", ""))
            known = {}
            for r, row in enumerate(rows):
                for c, ch in enumerate(row):
                    if ch.isalpha():
                        known[(r, c)] = ch.upper()
            record = bool(req.get("snapshots", False))
            res = FILLER.fill(puzzle, known=known, steps=int(req.get("steps", DEFAULT_STEPS)),
                              temperature=float(req.get("temperature", 0.0)), record=record,
                              remask_frac=float(req.get("remask_frac", 0.0)), samples=int(req.get("samples", 1)))

            def rows_of(letters: dict) -> list[str]:
                return ["".join("#" if (r, c) in puzzle.blocks else (letters.get((r, c)) or ".")
                                for c in range(puzzle.width)) for r in range(puzzle.height)]

            payload = {
                "grid": rows_of(res.letters),
                "confidence": {f"{r},{c}": round(p, 4) for (r, c), p in res.confidence.items()},
                "history": res.history,
                "elapsed_s": round(time.time() - t0, 2),
            }
            if record and res.snapshots is not None:
                payload["snapshots"] = [rows_of(snap) for snap in res.snapshots]
            self._json(200, payload)
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": repr(e)})

    def log_message(self, fmt, *args):  # quieter logs
        print(f"{self.address_string()} {fmt % args}")


def main() -> None:
    global FILLER
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--init-adapter", default=None, help="adapter the served adapter was trained on top of (merged first)")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--attn", default="eager")
    args = ap.parse_args()
    print(f"loading {args.base} (+{args.adapter})...", flush=True)
    FILLER = DiffusionFiller.load(args.base, args.adapter, attn=args.attn, init_adapter=args.init_adapter)
    FILLER.name = f"{args.base} + {args.init_adapter + ' + ' if args.init_adapter else ''}{args.adapter}"
    print(f"serving on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
