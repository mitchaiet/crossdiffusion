"""Command-line interface: solve, eval, compare, show, models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .agent import AgentConfig, CrosswordAgent
from .grid import Puzzle
from .llm import PRICING, LLMConfig, NebiusClient, load_dotenv
from .oracle import LLMOracle, SimulatedOracle
from .solver import SolverConfig

app = typer.Typer(add_completion=False, help="An LLM + constraint-solver agent for crossword puzzles.")
console = Console()

def _client(model: str | None, no_cache: bool, max_tokens: int | None, temperature: float | None,
            reasoning_effort: str | None = None) -> NebiusClient:
    cfg = LLMConfig.from_env(model=model, max_tokens=max_tokens, temperature=temperature,
                             reasoning_effort=reasoning_effort)
    cfg.use_cache = not no_cache
    return NebiusClient(cfg)


def _agent_config(k: int, rounds: int, threshold: float, repair: int, no_force: bool) -> AgentConfig:
    return AgentConfig(
        k=k, max_rounds=rounds, repair_cycles=repair, force_fill=not no_force,
        solver=SolverConfig(commit_threshold=threshold),
    )


@app.command()
def solve(
    path: Path = typer.Argument(..., help="Puzzle file (.json or .puz)"),
    model: str | None = typer.Option(None, help="Nebius model id (default: $CROSSWORD_MODEL or gpt-oss-120b)"),
    k: int = typer.Option(5, help="candidates requested per clue"),
    rounds: int = typer.Option(4, help="max propose/commit rounds"),
    threshold: float = typer.Option(0.8, help="posterior needed to commit an entry in round 1"),
    repair: int = typer.Option(1, help="judge/repair cycles"),
    no_force: bool = typer.Option(False, help="leave uncertain cells blank instead of guessing"),
    simulate: float | None = typer.Option(None, help="use a simulated oracle with this recall (no LLM)"),
    live: bool = typer.Option(False, help="animate the solve in the terminal"),
    out: Path | None = typer.Option(None, help="write the full trace JSON here"),
    no_cache: bool = typer.Option(False, help="bypass the on-disk LLM response cache"),
    max_tokens: int | None = typer.Option(None),
    temperature: float | None = typer.Option(None),
    only_wrong: bool = typer.Option(False, help="list only incorrect entries"),
    seed: int = typer.Option(0, help="seed for --simulate"),
    hybrid: bool = typer.Option(False, help="add the CrossDiffusion filler as a second oracle"),
    diffusion_url: str = typer.Option("http://100.106.68.91:8009", help="CrossDiffusion server for --hybrid"),
    index: int = typer.Option(0, help="which puzzle to take from a .jsonl corpus"),
) -> None:
    """Solve one puzzle and print the grid, entries and score."""
    from .evaluate import load_puzzle_set
    from .render import LiveView, print_result

    if path.suffix.lower() == ".jsonl":
        puzzle = load_puzzle_set([path], limit=index + 1)[index][1]
    else:
        puzzle = Puzzle.load(path)
    if simulate is not None:
        oracle = SimulatedOracle(puzzle, recall=simulate, seed=seed)
    else:
        oracle = LLMOracle(_client(model, no_cache, max_tokens, temperature))
        if hybrid:
            from .diffusion import CompositeOracle, DiffusionClient, DiffusionOracle

            oracle = CompositeOracle([oracle, DiffusionOracle(DiffusionClient(diffusion_url)).bind(puzzle)])
    cfg = _agent_config(k, rounds, threshold, repair, no_force)
    if live:
        with LiveView(console, puzzle) as view:
            result = CrosswordAgent(oracle, cfg, on_event=view).solve(puzzle)
        console.print()
    else:
        result = CrosswordAgent(oracle, cfg).solve(puzzle)
    print_result(console, result, only_wrong=only_wrong)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2))
        console.print(f"trace written to {out}")


@app.command("eval")
def eval_cmd(
    paths: list[Path] = typer.Argument(..., help="puzzle files or directories"),
    mode: str = typer.Option("agent", help="agent | oneshot | top1 | simulate | diffusion | hybrid"),
    model: str | None = typer.Option(None),
    name: str | None = typer.Option(None, help="run name (default: mode + model)"),
    out_dir: Path = typer.Option(Path("results")),
    k: int = typer.Option(5),
    rounds: int = typer.Option(4),
    threshold: float = typer.Option(0.8),
    repair: int = typer.Option(1),
    no_force: bool = typer.Option(False),
    recall: float = typer.Option(0.7, help="simulated oracle recall (mode=simulate)"),
    seed: int = typer.Option(0),
    no_cache: bool = typer.Option(False),
    max_tokens: int | None = typer.Option(None),
    temperature: float | None = typer.Option(None),
    reasoning_effort: str | None = typer.Option(None, help="low|medium|high (models that support it)"),
    concurrency: int = typer.Option(4, help="parallel LLM batches per puzzle"),
    save_traces: bool = typer.Option(True, help="write each puzzle's full trace to results/<run>/"),
    limit: int | None = typer.Option(None, help="use only the first N puzzles (handy for .jsonl corpora)"),
    diffusion_url: str = typer.Option("http://100.106.68.91:8009", help="CrossDiffusion server (modes diffusion/hybrid)"),
    diffusion_steps: int = typer.Option(12, help="denoising steps for the filler"),
    diffusion_min_conf: float = typer.Option(0.7, help="hybrid: ignore filler words below this confidence"),
    diffusion_trust: float = typer.Option(0.5, help="hybrid: multiplier on the filler's confidence"),
    hybrid_mode: str = typer.Option("inject", help="hybrid: inject (filler competes) | fallback (filler only fills blanks)"),
    snap: bool = typer.Option(False, help="diffusion: snap the filled grid onto the crossword vocabulary with BP"),
    diffusion_remask: float = typer.Option(0.0, help="diffusion: fraction of low-confidence cells re-masked each step"),
    diffusion_samples: int = typer.Option(1, help="diffusion: stochastic samples to vote over"),
) -> None:
    """Run a solver over a puzzle set and write a JSON + Markdown report."""
    from .baselines import solve_oneshot, solve_top1
    from .evaluate import aggregate, load_puzzle_set, run_eval, write_report

    items = load_puzzle_set(paths, limit)
    if not items:
        raise typer.BadParameter("no puzzles found")
    files = [label for label, _ in items]
    cfg = _agent_config(k, rounds, threshold, repair, no_force)
    needs_llm = mode in ("agent", "oneshot", "top1", "hybrid")
    client = _client(model, no_cache, max_tokens, temperature, reasoning_effort) if needs_llm else None
    model_id = client.config.model if client else ("simulated" if mode == "simulate" else "crossdiffusion")
    if mode == "hybrid":
        model_id += "+diffusion"
    run_name = name or f"{mode}__{model_id.replace('/', '_')}"
    dclient = None
    if mode in ("diffusion", "hybrid"):
        from .diffusion import DiffusionClient

        dclient = DiffusionClient(diffusion_url, steps=diffusion_steps, remask_frac=diffusion_remask, samples=diffusion_samples)

    def make_solver(puzzle: Puzzle):
        if mode == "simulate":
            oracle = SimulatedOracle(puzzle, recall=recall, seed=seed)
            return lambda p: CrosswordAgent(oracle, cfg).solve(p)
        if mode == "diffusion":
            from .diffusion import DiffusionSolver

            return lambda p: DiffusionSolver(dclient, snap=snap).solve(p)
        # fresh usage counters per puzzle so per-puzzle cost is attributable
        c = NebiusClient(client.config)
        oracle = LLMOracle(c, concurrency=concurrency)
        if mode == "agent":
            return lambda p: CrosswordAgent(oracle, cfg).solve(p)
        if mode == "hybrid":
            from .diffusion import CompositeOracle, DiffusionOracle

            combo = CompositeOracle([oracle, DiffusionOracle(dclient, min_conf=diffusion_min_conf, trust=diffusion_trust).bind(puzzle)],
                                    mode=hybrid_mode)
            return lambda p: CrosswordAgent(combo, cfg).solve(p)
        if mode == "oneshot":
            return lambda p: solve_oneshot(c, p)
        if mode == "top1":
            return lambda p: solve_top1(oracle, p, k=k)
        raise typer.BadParameter(f"unknown mode {mode}")

    table = Table(title=f"eval: {run_name}")
    for col in ("puzzle", "size", "letters", "words", "solved", "time", "cost"):
        table.add_column(col)
    rows: list[list[str]] = []

    trace_dir = out_dir / run_name

    def progress(m, result):
        if save_traces:
            trace_dir.mkdir(parents=True, exist_ok=True)
            fname = Path(m.file).name.replace("#", "_") + ".json"
            (trace_dir / fname).write_text(json.dumps(result.to_dict(), indent=1))
        cost = f"${m.cost_usd:.4f}" if m.cost_usd is not None else "-"
        console.print(
            f"[dim]{Path(m.file).name:>16}[/dim] {m.size:>6}  letters {m.letters_correct:>3}/{m.n_cells:<3} "
            f"words {m.words_correct:>2}/{m.n_slots:<2}  {'SOLVED' if m.solved else '      '}  {m.elapsed_s:5.1f}s  {cost}"
        )
        rows.append([Path(m.file).name, m.size, f"{m.letters_correct}/{m.n_cells}", f"{m.words_correct}/{m.n_slots}",
                     "yes" if m.solved else "no", f"{m.elapsed_s:.1f}s", cost])

    t0 = time.time()
    metrics = run_eval(items, make_solver, on_progress=progress)
    summary = aggregate(metrics)
    config = {"mode": mode, "model": model_id, "k": k, "rounds": rounds, "threshold": threshold,
              "repair": repair, "force_fill": not no_force, "recall": recall if mode == "simulate" else None,
              "puzzles": files, "wall_s": round(time.time() - t0, 1),
              "diffusion": {"url": diffusion_url, "steps": diffusion_steps, "min_conf": diffusion_min_conf,
                            "trust": diffusion_trust, "mode": hybrid_mode, "snap": snap,
                            "remask": diffusion_remask, "samples": diffusion_samples} if dclient else None}
    json_path, md_path = write_report(run_name, config, metrics, out_dir)
    console.print()
    console.print(
        f"[bold]{run_name}[/bold]: solved {summary['solved']}/{summary['puzzles']}  "
        f"letter acc {summary['letter_accuracy']:.1%}  word acc {summary['word_accuracy']:.1%}  "
        f"ECE {summary['calibration']['ece']}  cost {summary['cost_usd']}"
    )
    console.print(f"report: {md_path}  data: {json_path}")


@app.command()
def compare(results: list[Path] = typer.Argument(..., help="results/*.json files")) -> None:
    """Side-by-side summary of several eval runs."""
    table = Table(title="runs")
    for col in ("run", "model", "puzzles", "solved", "letter acc", "word acc", "fill", "ECE", "s/puzzle", "cost"):
        table.add_column(col)
    for path in results:
        d = json.loads(Path(path).read_text())
        s, c = d["summary"], d["config"]
        cost = f"${s['cost_usd']:.3f}" if s.get("cost_usd") is not None else "-"
        table.add_row(d["name"], str(c.get("model", "")).split("/")[-1], str(s["puzzles"]),
                      f"{s['solved']}/{s['puzzles']}", f"{s['letter_accuracy']:.1%}", f"{s['word_accuracy']:.1%}",
                      f"{s['fill_rate']:.0%}", str(s["calibration"].get("ece")), f"{s['mean_elapsed_s']:.1f}", cost)
    console.print(table)


@app.command()
def show(path: Path, reveal: bool = typer.Option(False, help="show the solution letters")) -> None:
    """Print a puzzle's grid and clues."""
    from .render import print_puzzle

    print_puzzle(console, Puzzle.load(path), reveal=reveal)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    diffusion_url: str | None = typer.Option(None, help="CrossDiffusion server (default $DIFFUSION_URL)"),
) -> None:
    """Launch the side-by-side demo web app (upload a screenshot, watch both solvers live)."""
    import os

    import uvicorn

    if diffusion_url:
        os.environ["DIFFUSION_URL"] = diffusion_url
    console.print(f"open http://{host}:{port}")
    uvicorn.run("webapp.server:app", host=host, port=port, log_level="warning")


@app.command()
def models() -> None:
    """List the Nebius models available to this key, with known pricing."""
    load_dotenv()
    client = _client(None, True, None, None)
    table = Table(title="Nebius Token Factory models")
    for col in ("model", "$/1M in", "$/1M out"):
        table.add_column(col)
    for m in sorted(x.id for x in client._client.models.list().data):
        price = PRICING.get(m)
        table.add_row(m, f"{price[0]:.2f}" if price else "?", f"{price[1]:.2f}" if price else "?")
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
