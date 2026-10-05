"""Terminal rendering with rich: grids, answer tables, and a live view for demos."""

from __future__ import annotations

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import SolveResult
from .grid import Grid, Puzzle


def grid_text(grid: Grid, puzzle: Puzzle | None = None, check: bool = True, highlight: set | None = None) -> Text:
    """Render a grid. Green = matches solution, red = differs, dim dot = empty."""
    puzzle = puzzle or grid.puzzle
    t = Text()
    for r in range(puzzle.height):
        for c in range(puzzle.width):
            cell = (r, c)
            if cell in puzzle.blocks:
                t.append("██", style="grey23")
            else:
                ch = grid.letters[cell]
                if ch is None:
                    t.append("· ", style="grey50")
                else:
                    style = "bold"
                    if check and puzzle.solution:
                        style = "bold green" if puzzle.solution[cell] == ch else "bold red"
                    if highlight and cell in highlight:
                        style += " reverse"
                    t.append(ch + " ", style=style)
        t.append("\n")
    return t


def answers_table(result: SolveResult, only_wrong: bool = False) -> Table:
    p = result.puzzle
    table = Table(title="Entries", show_lines=False, expand=False)
    table.add_column("Slot", justify="right")
    table.add_column("Clue")
    table.add_column("Answer")
    table.add_column("Conf", justify="right")
    if p.solution:
        table.add_column("Truth")
    for s in p.slots:
        w = result.grid.word(s) or "—"
        truth = p.solution_word(s) if p.solution else None
        ok = truth is None or w == truth
        if only_wrong and ok:
            continue
        style = "" if truth is None else ("green" if ok else "red")
        row = [s.id, s.clue[:60], Text(w, style=style), f"{result.confidences.get(s.id, 0):.2f}"]
        if p.solution:
            row.append(truth or "")
        table.add_row(*row)
    return table


def score_line(result: SolveResult) -> Text:
    p = result.puzzle
    if not p.solution:
        return Text(f"filled {result.grid.filled_fraction():.0%} of cells in {result.elapsed_s:.1f}s")
    cells = p.white_cells
    correct = sum(1 for c in cells if result.grid.letters[c] == p.solution[c])
    words = sum(1 for s in p.slots if result.grid.word(s) == p.solution_word(s))
    solved = correct == len(cells)
    t = Text()
    t.append("SOLVED  " if solved else "partial ", style="bold green" if solved else "bold yellow")
    t.append(f"letters {correct}/{len(cells)} ({correct / len(cells):.0%})  words {words}/{len(p.slots)}  "
             f"rounds {result.rounds}  {result.elapsed_s:.1f}s")
    if result.usage:
        u = result.usage
        t.append(f"  LLM calls {u.get('calls', 0)} (+{u.get('cache_hits', 0)} cached)  "
                 f"tokens {u.get('prompt_tokens', 0)}/{u.get('completion_tokens', 0)}", style="dim")
    return t


def print_result(console: Console, result: SolveResult, show_answers: bool = True, only_wrong: bool = False) -> None:
    console.print(Panel(grid_text(result.grid), title=result.puzzle.title or "grid", expand=False))
    console.print(score_line(result))
    if show_answers:
        console.print(answers_table(result, only_wrong=only_wrong))


def print_puzzle(console: Console, puzzle: Puzzle, reveal: bool = False) -> None:
    g = Grid(puzzle)
    if reveal and puzzle.solution:
        for c in puzzle.white_cells:
            g.letters[c] = puzzle.solution[c]
    console.print(Panel(grid_text(g, check=False), title=puzzle.title, expand=False))
    for direction, name in (("A", "Across"), ("D", "Down")):
        console.print(f"[bold]{name}[/bold]")
        for s in puzzle.slots:
            if s.direction == direction:
                console.print(f"  {s.number:>3}. {s.clue} ({s.length})")


class LiveView:
    """`on_event` callback that animates the solve in the terminal."""

    def __init__(self, console: Console, puzzle: Puzzle, max_log: int = 12) -> None:
        self.console = console
        self.puzzle = puzzle
        self.log: list[Text] = []
        self.max_log = max_log
        self.live = Live(console=console, refresh_per_second=8, transient=False)
        self.grid = Grid(puzzle)
        self.last_cells: set = set()

    def __enter__(self) -> LiveView:
        self.live.__enter__()
        self._refresh()
        return self

    def __exit__(self, *exc) -> None:
        self._refresh()
        self.live.__exit__(*exc)

    def _refresh(self) -> None:
        header = Text(f"{self.puzzle.title}  ({self.puzzle.width}x{self.puzzle.height})", style="bold")
        panel = Panel(grid_text(self.grid, highlight=self.last_cells), expand=False)
        body = Group(header, panel, *self.log[-self.max_log :])
        self.live.update(body)

    def __call__(self, event: dict, grid: Grid) -> None:
        self.grid = grid.copy()
        kind = event.get("type")
        if kind == "stage":
            msg = Text(event["text"], style="cyan")
            self.last_cells = set()
        elif kind == "propose":
            msg = Text(f"round {event['round']}: asked the LLM about {event['queries']} clues, "
                       f"{event['new']} new candidates", style="cyan")
            self.last_cells = set()
        elif kind == "commit":
            slot = self.puzzle.slot_by_id[event["slot"]]
            self.last_cells = set(slot.cells)
            msg = Text(f"  commit {event['slot']:>4} = {event['word']:<14} p={event['p']:.2f}", style="green")
        elif kind == "round":
            msg = Text(f"round {event['round']}: {event['commits']} commits, grid {event['filled']:.0%} filled "
                       f"(threshold {event['threshold']:.2f})", style="bold")
        elif kind == "backtrack":
            msg = Text(f"search: {event['nodes']} nodes, grid {event['filled']:.0%} filled", style="magenta")
            self.last_cells = set()
        elif kind == "reject":
            msg = Text(f"  judge rejected {event['slot']} = {event['word']} -> {event.get('alternative')}", style="red")
        elif kind == "judge":
            msg = Text(f"judge: checked {event['checked']} weak entries, rejected {event['rejected']}", style="yellow")
        elif kind == "force_fill":
            msg = Text(f"  best guess {event['slot']} = {event['word']}", style="yellow")
        elif kind == "done":
            msg = Text(f"done in {event['elapsed_s']}s, grid {event['filled']:.0%} filled", style="bold green")
        else:
            msg = Text(str(event))
        self.log.append(msg)
        self._refresh()
