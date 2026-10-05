"""Turn a crossword into model tokens for masked-diffusion training and sampling.

Sequence layout (prefix-LM):

    <bos> PROMPT_TEXT  |  CANVAS
    ------ causal ---- | -- bidirectional --

PROMPT_TEXT is ordinary text: size, title, then every clue with its length.
CANVAS is one token per cell in row-major order, with a row-break token after each
row, so column = position mod (width + 1). Cell tokens are ordinary vocabulary items
chosen so each cell is exactly one token:

    letter  " A" .. " Z"      black  " #"      masked  " _"      row break  "\\n"

No vocabulary surgery is needed; at canvas positions the loss and the sampler only
consider the 26 letter token ids.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class CanvasVocab:
    letter_ids: list[int]  # 26 ids, index = ord(ch) - 65
    black_id: int
    mask_id: int
    row_id: int

    @classmethod
    def from_tokenizer(cls, tok) -> CanvasVocab:
        def single(text: str) -> int:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{text!r} is not a single token: {ids}")
            return ids[0]

        return cls(
            letter_ids=[single(" " + ch) for ch in LETTERS],
            black_id=single(" #"),
            mask_id=single(" _"),
            row_id=single("\n"),
        )

    def letter_index(self) -> dict[int, int]:
        return {tid: i for i, tid in enumerate(self.letter_ids)}


def prompt_text(puzzle) -> str:
    """Human-readable clue list; the causal half of the sequence."""
    lines = [f"Crossword {puzzle.width}x{puzzle.height}."]
    if puzzle.title:
        lines.append(f"Title: {puzzle.title}")
    across = [s for s in puzzle.slots if s.direction == "A"]
    down = [s for s in puzzle.slots if s.direction == "D"]
    lines.append("Across: " + " ".join(f"{s.number}. {s.clue} ({s.length})" for s in across))
    lines.append("Down: " + " ".join(f"{s.number}. {s.clue} ({s.length})" for s in down))
    lines.append("Grid:")
    return "\n".join(lines) + "\n"


def canvas_ids(puzzle, vocab: CanvasVocab, letters: dict | None = None) -> tuple[list[int], list[tuple[int, int] | None]]:
    """Token ids for the canvas plus, per position, the cell it encodes (None for row breaks).

    `letters` maps cell -> letter or None (masked). Defaults to the puzzle solution.
    """
    if letters is None:
        letters = puzzle.solution or {}
    ids: list[int] = []
    cells: list[tuple[int, int] | None] = []
    for r in range(puzzle.height):
        for c in range(puzzle.width):
            cell = (r, c)
            if cell in puzzle.blocks:
                ids.append(vocab.black_id)
            else:
                ch = letters.get(cell)
                ids.append(vocab.letter_ids[ord(ch) - 65] if ch else vocab.mask_id)
            cells.append(cell)
        ids.append(vocab.row_id)
        cells.append(None)
    return ids, cells


def mask_cells(puzzle, rng: random.Random, t: float | None = None, t1_frac: float = 0.15,
               noise: float = 0.0) -> tuple[dict, float]:
    """Masked-diffusion corruption: mask each white cell independently with prob t.

    t1_frac  share of examples that are fully masked (inference starts from a blank grid)
    noise    fraction of *visible* letters replaced by a wrong letter, so the model learns not to
             trust every crossing it sees (at inference, revealed letters can be wrong)
    """
    if t is None:
        t = 1.0 if rng.random() < t1_frac else rng.random()
    t = min(1.0, max(0.02, t))
    letters = {}
    n_masked = 0
    for cell in puzzle.white_cells:
        if rng.random() < t:
            letters[cell] = None
            n_masked += 1
        else:
            truth = puzzle.solution[cell]
            if noise > 0 and rng.random() < noise:
                letters[cell] = rng.choice(LETTERS.replace(truth, ""))
            else:
                letters[cell] = truth
    if n_masked == 0:  # guarantee at least one target
        cell = rng.choice(puzzle.white_cells)
        letters[cell] = None
    return letters, t


@dataclass
class Example:
    input_ids: list[int]
    prompt_len: int  # tokens before the canvas (causal region)
    labels: list[int]  # letter index 0-25 at masked canvas positions, -100 elsewhere
    t: float
    n_white: int


def build_example(puzzle, tok, vocab: CanvasVocab, rng: random.Random, t: float | None = None, max_prompt_tokens: int = 3072,
                  t1_frac: float = 0.15, noise: float = 0.0) -> Example:
    prompt_ids = tok.encode(prompt_text(puzzle), add_special_tokens=True)[:max_prompt_tokens]
    letters, t = mask_cells(puzzle, rng, t, t1_frac=t1_frac, noise=noise)
    cids, cells = canvas_ids(puzzle, vocab, letters)
    labels = []
    for tid, cell in zip(cids, cells, strict=True):
        if cell is not None and tid == vocab.mask_id:
            labels.append(ord(puzzle.solution[cell]) - 65)
        else:
            labels.append(-100)
    return Example(
        input_ids=prompt_ids + cids,
        prompt_len=len(prompt_ids),
        labels=[-100] * len(prompt_ids) + labels,
        t=t,
        n_white=len(puzzle.white_cells),
    )


def prefix_lm_mask(seq_len: int, prompt_len: int, device=None, dtype=None):
    """4D additive attention mask: causal over the prompt, bidirectional over the canvas,
    canvas attends to the whole prompt, prompt never attends to the canvas."""
    import torch

    allowed = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    allowed[prompt_len:, prompt_len:] = True
    mask = torch.zeros(seq_len, seq_len, dtype=dtype or torch.float32, device=device)
    mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
    return mask[None, None]  # (1, 1, q, k)
